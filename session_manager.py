import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from cloak_config import RETRIEVE_K
from ingestion_pipeline import create_ephemeral_vectorstore, load_chunks_from_vectorstore

IDLE_SECONDS = 30 * 60
SWEEP_INTERVAL_SECONDS = 60


def _bm25_tokens(text):
    return re.findall(r"[a-z0-9][a-z0-9_./+-]*", (text or "").lower())


class SessionState:
    def __init__(self, session_id):
        self.session_id = session_id
        self.temp_dir = tempfile.mkdtemp(prefix=f"cloak_{session_id[:8]}_")
        collection_name = "s" + session_id.replace("-", "")[:32]
        self.collection_name = collection_name
        self.vectorstore, self._chroma_client = create_ephemeral_vectorstore(
            collection_name=collection_name
        )
        self.chunks = []
        self.chat_history = []
        self.hybrid_retriever = None
        self.bm25_retriever = None
        self.last_seen = time.time()
        self.in_use = 0
        self._close_requested = False
        self.lock = threading.Lock()
        self.rebuild_retrievers()

    def rebuild_retrievers(self):
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVE_K},
        )
        usable = [chunk for chunk in self.chunks if (chunk.page_content or "").strip()]
        self.bm25_retriever = None
        if usable:
            try:
                bm25_retriever = BM25Retriever.from_documents(
                    usable,
                    preprocess_func=_bm25_tokens,
                )
                bm25_retriever.k = RETRIEVE_K
                self.bm25_retriever = bm25_retriever
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.55, 0.45],
                )
                return
            except Exception as exc:
                print(f"BM25 rebuild failed, using vector search only: {exc}")
        self.hybrid_retriever = vector_retriever

    def destroy(self):
        try:
            if self._chroma_client is not None:
                self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.chunks = []
        self.chat_history = []
        self.hybrid_retriever = None
        self.bm25_retriever = None
        self.vectorstore = None
        self._chroma_client = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class SessionManager:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sweeper = None

    def start_sweeper(self):
        if self._sweeper and self._sweeper.is_alive():
            return
        self._stop.clear()
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="cloak-session-sweeper",
            daemon=True,
        )
        self._sweeper.start()

    def shutdown(self):
        self._stop.set()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.destroy()
        print("All sessions wiped on shutdown.")

    def get_or_create(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionState(session_id)
                self._sessions[session_id] = session
            session.last_seen = time.time()
            return session

    @contextmanager
    def session_scope(self, session_id):
        session = self.get_or_create(session_id)
        with session.lock:
            session.in_use += 1
            session.last_seen = time.time()
            try:
                yield session
            finally:
                session.in_use -= 1
                session.last_seen = time.time()
                if session._close_requested and session.in_use == 0:
                    self.close(session_id)

    def close(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session._close_requested = True
            if session.in_use > 0:
                print(f"Session {session_id} close deferred until in-flight request finishes.")
                return True
            self._sessions.pop(session_id, None)
        session.destroy()
        print(f"Closed session {session_id} and wiped temp files.")
        return True

    def _sweep_loop(self):
        while not self._stop.wait(SWEEP_INTERVAL_SECONDS):
            self._sweep_idle()

    def _sweep_idle(self):
        now = time.time()
        stale = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if session.in_use == 0 and (now - session.last_seen) > IDLE_SECONDS:
                    stale.append(session_id)
        for session_id in stale:
            print(f"Idle timeout: closing session {session_id}")
            self.close(session_id)


session_manager = SessionManager()
