const $ = (id) => document.getElementById(id);

if (window.cloakDesktop) document.body.classList.add("desktop");

const state = {
  user: null,
  chats: [],
  chat: null,
  pdfs: [],
  models: [],
  model: "",
  routeMode: "auto",
  busy: false,
  abort: null,
  uploadTimer: null,
  drafts: {},
  live: {},
  streamChatId: null,
  authRegister: false,
  showPassword: false,
};

function api(path, options = {}) {
  return fetch(path, {
    credentials: "include",
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(options.headers || {}),
    },
  });
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || data.message || response.statusText;
  } catch {
    return response.statusText;
  }
}

function show(id, on) {
  $(id).hidden = !on;
}

function avatarLetter(name) {
  return (name || "C").trim().slice(0, 1).toUpperCase();
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMath(tex, display) {
  const source = String(tex || "").trim();
  if (!source) return "";
  if (window.katex) {
    try {
      return window.katex.renderToString(source, {
        displayMode: Boolean(display),
        throwOnError: false,
        output: "html",
      });
    } catch {
      /* fall through */
    }
  }
  return escapeHtml(display ? `\\[${source}\\]` : `\\(${source}\\)`);
}

function formatProse(text) {
  const slots = [];
  const replaced = String(text || "").replace(
    /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)/g,
    (_match, dollar, bracket, paren) => {
      const display = dollar != null || bracket != null;
      const tex = dollar ?? bracket ?? paren;
      const key = `\u0000MATH${slots.length}\u0000`;
      slots.push(renderMath(tex, display));
      return key;
    }
  );
  let html = escapeHtml(replaced)
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^\d+\. (.+)$/gm, "<li data-ol>$1</li>")
    .replace(/^(?:- |\* )(.+)$/gm, "<li>$1</li>")
    .replace(/(?:<li data-ol>.*<\/li>\n?)+/g, (block) => `<ol>${block.replace(/ data-ol/g, "")}</ol>`)
    .replace(/(?:<li>.*<\/li>\n?)+/g, (block) => `<ul>${block}</ul>`)
    .replace(/\n/g, "<br>");
  slots.forEach((math, index) => {
    html = html.replace(`\u0000MATH${index}\u0000`, math);
  });
  return html;
}

function renderMarkdown(text) {
  const parts = String(text || "").split(/```/);
  return parts
    .map((part, index) => {
      if (index % 2 === 1) {
        const nl = part.indexOf("\n");
        const lang = nl === -1 ? "" : part.slice(0, nl).trim();
        const code = nl === -1 ? part : part.slice(nl + 1);
        return `<pre><code class="${escapeHtml(lang)}">${escapeHtml(code)}</code></pre>`;
      }
      return formatProse(part);
    })
    .join("");
}

function setAuthError(message) {
  const node = $("auth-error");
  node.hidden = !message;
  node.textContent = message || "";
}

async function boot() {
  const me = await api("/auth/me");
  if (me.ok) {
    const data = await me.json();
    state.user = data.user;
    await enterApp();
    return;
  }
  show("login-view", true);
  show("app-view", false);
}

function setPasswordVisible(on) {
  state.showPassword = Boolean(on);
  const type = state.showPassword ? "text" : "password";
  $("auth-pass").type = type;
  $("auth-confirm").type = type;
  document.querySelectorAll(".pass-toggle").forEach((button) => {
    button.setAttribute("aria-pressed", state.showPassword ? "true" : "false");
    button.setAttribute("aria-label", state.showPassword ? "Hide password" : "Show password");
    button.title = state.showPassword ? "Hide password" : "Show password";
    const eye = button.querySelector(".icon-eye");
    const off = button.querySelector(".icon-eye-off");
    if (eye) eye.hidden = state.showPassword;
    if (off) off.hidden = !state.showPassword;
  });
}

function setAuthMode(register) {
  state.authRegister = Boolean(register);
  show("auth-confirm-row", state.authRegister);
  $("login-sub").textContent = state.authRegister
    ? "Create a local account on this machine."
    : "Private local chat. Sign in to this machine.";
  $("auth-login").textContent = state.authRegister ? "Create account" : "Log in";
  $("auth-register").textContent = state.authRegister ? "Back to log in" : "Create account";
  $("auth-pass").autocomplete = state.authRegister ? "new-password" : "current-password";
  $("auth-confirm").required = state.authRegister;
  if (!state.authRegister) $("auth-confirm").value = "";
  setAuthError("");
}

async function submitAuth() {
  setAuthError("");
  const username = $("auth-user").value.trim();
  const password = $("auth-pass").value;
  const confirm = $("auth-confirm").value;
  if (state.authRegister) {
    if (!confirm) {
      setAuthError("Confirm your password.");
      $("auth-confirm").focus();
      return;
    }
    if (password !== confirm) {
      setAuthError("Passwords do not match.");
      $("auth-confirm").focus();
      return;
    }
  }
  const path = state.authRegister ? "/auth/register" : "/auth/login";
  const body = { username, password };
  if (state.authRegister) body.confirm_password = confirm;
  const response = await api(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    setAuthError(await readError(response));
    return;
  }
  const data = await response.json();
  state.user = data.user;
  setAuthMode(false);
  setPasswordVisible(false);
  $("auth-form").reset();
  await enterApp();
}

async function enterApp() {
  show("login-view", false);
  show("app-view", true);
  $("user-name").textContent = state.user.username;
  $("user-avatar").textContent = avatarLetter(state.user.username);
  await Promise.all([loadModels(), loadPdfs(), loadChats()]);
  if (!state.chat) {
    if (state.chats.length) await openChat(state.chats[0].id);
    else await createChat();
  }
}

async function loadModels() {
  try {
    const response = await api("/models");
    if (!response.ok) return;
    const data = await response.json();
    state.models = data.models || [];
    state.model = data.default || state.models[0] || "";
    $("model-label").textContent = shortModel(state.model);
  } catch {
    $("model-label").textContent = "Offline";
  }
}

function shortModel(name) {
  if (!name) return "Model";
  return name.replace(":latest", "");
}

async function loadPdfs() {
  const response = await api("/pdfs");
  if (!response.ok) return;
  const data = await response.json();
  state.pdfs = data.documents || [];
  renderPdfs();
}

async function loadChats() {
  const response = await api("/chats");
  if (!response.ok) return;
  const data = await response.json();
  state.chats = data.chats || [];
  renderChatList();
}

function isOpenChat(id) {
  return Boolean(state.chat && state.chat.id === id);
}

function rememberChat(chat) {
  if (chat && chat.id) state.live[chat.id] = chat;
  return chat;
}

function saveDraft() {
  if (state.chat) state.drafts[state.chat.id] = $("prompt").value;
}

function loadDraft() {
  $("prompt").value = (state.chat && state.drafts[state.chat.id]) || "";
  resizePrompt();
}

function setSendMode(stopping) {
  $("send-btn").classList.toggle("stop", stopping);
  $("send-btn").title = stopping ? "Stop" : "Send";
}

async function createChat() {
  saveDraft();
  const response = await api("/chats", { method: "POST" });
  if (!response.ok) return;
  const data = await response.json();
  state.chat = rememberChat(data.chat);
  await loadChats();
  loadDraft();
  renderThread();
  renderPdfs();
}

async function openChat(id) {
  if (isOpenChat(id)) return;
  saveDraft();
  if (state.live[id] && state.streamChatId === id) {
    state.chat = state.live[id];
    renderChatList();
    loadDraft();
    renderThread();
    renderPdfs();
    return;
  }
  const response = await api(`/chats/${id}`);
  if (!response.ok) return;
  const data = await response.json();
  const chat = data.chat;
  const live = state.live[id];
  if (live && (live.messages || []).length > (chat.messages || []).length) {
    chat.messages = live.messages;
  }
  state.chat = rememberChat(chat);
  renderChatList();
  loadDraft();
  renderThread();
  renderPdfs();
}

function renderChatList() {
  const filter = ($("chat-filter").value || "").toLowerCase();
  const root = $("chat-list");
  root.innerHTML = "";
  for (const chat of state.chats) {
    const title = chat.title || "New chat";
    if (filter && !title.toLowerCase().includes(filter)) continue;
    const row = document.createElement("div");
    row.className = "chat-item" + (state.chat && state.chat.id === chat.id ? " active" : "");
    row.innerHTML = `
      <button type="button" class="chat-open"><span>${escapeHtml(title)}</span></button>
      <button type="button" class="chat-delete" title="Delete chat" aria-label="Delete chat">
        <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M6 7h12v2H6zm2 3h2v8H8zm6 0h2v8h-2zM9 4h6l1 2H8z"/></svg>
      </button>`;
    row.querySelector(".chat-open").addEventListener("click", () => openChat(chat.id));
    row.querySelector(".chat-delete").addEventListener("click", (event) => {
      event.stopPropagation();
      askDeleteChat(chat);
    });
    root.appendChild(row);
  }
}

let pendingDelete = null;

function askDeleteChat(chat) {
  pendingDelete = chat;
  $("confirm-title").textContent = `“${chat.title || "New chat"}”`;
  show("confirm-dialog", true);
}

function closeConfirm() {
  pendingDelete = null;
  show("confirm-dialog", false);
}

async function confirmDeleteChat() {
  const chat = pendingDelete;
  if (!chat) return;
  const id = chat.id;
  const wasOpen = state.chat && state.chat.id === id;
  closeConfirm();
  const response = await api(`/chats/${id}`, { method: "DELETE" });
  if (!response.ok) {
    alert(await readError(response));
    return;
  }
  if (wasOpen) state.chat = null;
  await loadChats();
  if (wasOpen) {
    if (state.chats.length) await openChat(state.chats[0].id);
    else {
      state.chat = null;
      renderChatList();
      renderThread();
      renderPdfs();
    }
  }
}

function renderThread() {
  const messages = (state.chat && state.chat.messages) || [];
  const empty = messages.length === 0;
  show("empty", empty);
  show("thread", !empty);
  const thread = $("thread");
  thread.innerHTML = "";
  for (const message of messages) {
    thread.appendChild(messageNode(message));
  }
  thread.scrollTop = thread.scrollHeight;
}

function messageNode(message) {
  const wrap = document.createElement("div");
  if (message.role === "user") {
    wrap.className = "turn user";
    wrap.innerHTML = `<div class="user-bubble">${escapeHtml(message.content)}</div>`;
    return wrap;
  }
  wrap.className = "turn";
  wrap.dataset.role = "assistant";
  const cites = (message.citations || [])
    .map((item) => `<span class="cite">${escapeHtml(item.source || "PDF")}${item.page ? " p." + item.page : ""}</span>`)
    .join("");
  const thinking =
    message.stage && !message.content
      ? `<div class="thinking">${escapeHtml(message.stage)}</div>`
      : "";
  const thoughts = message.reasoning
    ? `<details class="thoughts"><summary>Thoughts</summary><div>${escapeHtml(message.reasoning)}</div></details>`
    : "";
  wrap.innerHTML = `
    <div class="assistant">
      ${thinking}
      ${thoughts}
      <div class="body">${renderMarkdown(message.content)}</div>
      ${cites ? `<div class="citations">${cites}</div>` : ""}
    </div>`;
  return wrap;
}

function patchLastAssistant(message, chatId) {
  if (!isOpenChat(chatId)) return;
  const thread = $("thread");
  const last = thread.lastElementChild;
  if (last && last.dataset.role === "assistant") {
    const next = messageNode(message);
    thread.replaceChild(next, last);
    thread.scrollTop = thread.scrollHeight;
    return;
  }
  renderThread();
}

function renderPdfs() {
  const root = $("pdf-list");
  root.innerHTML = "";
  const attached = new Set((state.chat && state.chat.content_hashes) || []);
  if (!state.pdfs.length) {
    root.innerHTML = `<p class="hint">No PDFs yet.</p>`;
    return;
  }
  for (const doc of state.pdfs) {
    const row = document.createElement("label");
    row.className = "pdf-row";
    const checked = attached.has(doc.content_hash) ? "checked" : "";
    row.innerHTML = `
      <input type="checkbox" ${checked} />
      <span title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
      <button type="button" class="text-btn">Remove</button>
    `;
    const box = row.querySelector("input");
    box.addEventListener("change", () => togglePdf(doc.content_hash, box.checked));
    row.querySelector("button").addEventListener("click", (event) => {
      event.preventDefault();
      deletePdf(doc.content_hash);
    });
    root.appendChild(row);
  }
}

async function togglePdf(contentHash, on) {
  if (!state.chat) return;
  const hashes = new Set(state.chat.content_hashes || []);
  if (on) hashes.add(contentHash);
  else hashes.delete(contentHash);
  const response = await api(`/chats/${state.chat.id}`, {
    method: "PATCH",
    body: JSON.stringify({ content_hashes: [...hashes] }),
  });
  if (response.ok) {
    const data = await response.json();
    state.chat = data.chat;
    renderPdfs();
  }
}

async function deletePdf(contentHash) {
  const response = await api(`/pdfs/${contentHash}`, { method: "DELETE" });
  if (response.ok) await loadPdfs();
}

const UPLOAD_STEPS = [
  "Uploading file…",
  "Reading pages…",
  "Extracting text…",
  "Building search index…",
  "Almost done…",
];

function showUploadOverlay(filename) {
  $("upload-file").textContent = filename || "PDF";
  $("upload-title").textContent = "Adding PDF";
  $("upload-status").textContent = UPLOAD_STEPS[0];
  show("upload-overlay", true);
  $("upload-pdf").disabled = true;
  $("attach-btn").disabled = true;
  let step = 0;
  state.uploadTimer = setInterval(() => {
    step = Math.min(step + 1, UPLOAD_STEPS.length - 1);
    $("upload-status").textContent = UPLOAD_STEPS[step];
  }, 2800);
}

function hideUploadOverlay() {
  if (state.uploadTimer) {
    clearInterval(state.uploadTimer);
    state.uploadTimer = null;
  }
  show("upload-overlay", false);
  $("upload-pdf").disabled = false;
  $("attach-btn").disabled = false;
}

async function uploadPdf(file) {
  if (!file) return;
  show("pdf-drawer", true);
  showUploadOverlay(file.name);
  try {
    const form = new FormData();
    form.append("file", file, file.name);
    const response = await api("/pdfs", { method: "POST", body: form });
    if (!response.ok) throw new Error(await readError(response));
    $("upload-status").textContent = "Attached to this chat.";
    const data = await response.json();
    await loadPdfs();
    if (data.content_hash) await togglePdf(data.content_hash, true);
  } catch (err) {
    hideUploadOverlay();
    alert(err.message || "Could not add that PDF.");
    return;
  }
  hideUploadOverlay();
}

function resizePrompt() {
  const node = $("prompt");
  node.style.height = "auto";
  node.style.height = Math.min(node.scrollHeight, 160) + "px";
  $("send-btn").hidden = !(node.value.trim() || state.busy);
}

async function sendPrompt() {
  const query = $("prompt").value.trim();
  if (!query || state.busy) return;
  if (!state.chat) await createChat();
  const chat = state.chat;
  if (!chat) return;
  $("prompt").value = "";
  state.drafts[chat.id] = "";
  resizePrompt();
  chat.messages = chat.messages || [];
  chat.messages.push({ role: "user", content: query });
  rememberChat(chat);
  if (isOpenChat(chat.id)) renderThread();
  await streamAnswer(query, chat);
}

async function streamAnswer(query, chat) {
  const chatId = chat.id;
  state.busy = true;
  state.streamChatId = chatId;
  $("send-btn").hidden = false;
  setSendMode(true);
  const assistant = { role: "assistant", content: "", citations: [], stage: "Thinking…", reasoning: "" };
  chat.messages.push(assistant);
  if (isOpenChat(chatId)) renderThread();
  state.abort = new AbortController();
  try {
    const response = await api("/query/stream", {
      method: "POST",
      body: JSON.stringify({
        query,
        chat_id: chatId,
        model: state.model,
        route_mode: state.routeMode,
        content_hashes: chat.content_hashes || [],
      }),
      signal: state.abort.signal,
    });
    if (!response.ok) throw new Error(await readError(response));
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      for (const chunk of chunks) {
        const line = chunk.split("\n").find((item) => item.startsWith("data:"));
        if (!line) continue;
        const event = JSON.parse(line.slice(5).trim() || "{}");
        if (event.type === "stage") {
          const raw = event.detail || event.label || "Thinking…";
          assistant.stage = /^(assistant|responding)$/i.test(String(raw).trim())
            ? "Thinking…"
            : raw;
          patchLastAssistant(assistant, chatId);
        }
        if (event.type === "token") {
          assistant.stage = "";
          assistant.content = event.answer || assistant.content + (event.text || "");
          patchLastAssistant(assistant, chatId);
        }
        if (event.type === "complete") {
          assistant.stage = "";
          assistant.content = event.answer || assistant.content;
          assistant.citations = event.citations || [];
          assistant.reasoning = event.reasoning || "";
          patchLastAssistant(assistant, chatId);
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      assistant.stage = "";
      assistant.content = assistant.content || "Could not reach Cloak.";
      if (isOpenChat(chatId)) renderThread();
    } else if (!assistant.content) {
      assistant.stage = "";
      assistant.content = "Stopped.";
      if (isOpenChat(chatId)) renderThread();
    }
  } finally {
    state.busy = false;
    state.abort = null;
    state.streamChatId = null;
    setSendMode(false);
    resizePrompt();
    await loadChats();
  }
}

function cancelSend() {
  if (state.abort) state.abort.abort();
  api("/query/cancel", { method: "POST" });
}

function placeMenu(menu, around) {
  menu.style.maxHeight = "none";
  menu.hidden = false;
  const box = around.getBoundingClientRect();
  const pad = 8;
  const width = Math.max(menu.offsetWidth, 200);
  const height = menu.offsetHeight;
  let left = box.left;
  if (box.left + width > window.innerWidth - pad) {
    left = box.right - width;
  }
  left = Math.min(Math.max(pad, left), window.innerWidth - width - pad);

  const gap = 6;
  const spaceBelow = window.innerHeight - box.bottom - pad;
  const spaceAbove = box.top - pad;
  const openUp = spaceBelow < height + gap && spaceAbove > spaceBelow;
  let top;
  if (openUp) {
    const maxH = Math.max(120, spaceAbove - gap);
    menu.style.maxHeight = `${maxH}px`;
    top = Math.max(pad, box.top - Math.min(height, maxH) - gap);
  } else {
    const maxH = Math.max(120, spaceBelow - gap);
    menu.style.maxHeight = `${maxH}px`;
    top = box.bottom + gap;
  }
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
}

function renderModelMenu() {
  const menu = $("model-menu");
  menu.innerHTML = "";
  for (const name of state.models) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = shortModel(name);
    if (name === state.model) button.className = "active";
    button.addEventListener("click", () => {
      state.model = name;
      $("model-label").textContent = shortModel(name);
      menu.hidden = true;
    });
    menu.appendChild(button);
  }
}

let mic = { rec: null, listening: false };

function speechEngine() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function setMicUi(on, message) {
  $("mic-btn").classList.toggle("listening", on);
  $("mic-btn").title = on ? "Stop listening" : "Voice";
  $("prompt").placeholder = message || (on ? "Listening…" : "Ask Cloak");
}

function stopMic() {
  mic.listening = false;
  if (mic.rec) {
    try {
      mic.rec.stop();
    } catch {
      /* already stopped */
    }
  }
  mic.rec = null;
  setMicUi(false);
}

async function toggleMic() {
  if (mic.listening) {
    stopMic();
    return;
  }
  const Speech = speechEngine();
  if (!Speech) {
    setMicUi(false, "Voice needs Chrome or Edge");
    setTimeout(() => setMicUi(false), 2800);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((track) => track.stop());
  } catch {
    setMicUi(false, "Microphone permission denied");
    setTimeout(() => setMicUi(false), 2800);
    return;
  }
  const rec = new Speech();
  rec.lang = navigator.language || "en-US";
  rec.interimResults = true;
  rec.continuous = true;
  rec.onresult = (event) => {
    const result = event.results[event.results.length - 1];
    if (!result || !result.isFinal) return;
    const said = (result[0] && result[0].transcript || "").trim();
    if (!said) return;
    const node = $("prompt");
    node.value = node.value.trim() ? `${node.value.trim()} ${said}` : said;
    resizePrompt();
  };
  rec.onerror = (event) => {
    stopMic();
    const desktop = Boolean(window.cloakDesktop);
    const messages = {
      "not-allowed": "Microphone blocked",
      "audio-capture": "No microphone found",
      "no-speech": "No speech heard",
      network: desktop
        ? "Voice needs Chrome at http://127.0.0.1:8001"
        : "Speech service unavailable",
      "service-not-allowed": "Speech not allowed here",
    };
    setMicUi(false, messages[event.error] || `Mic error: ${event.error}`);
    setTimeout(() => setMicUi(false), 4000);
  };
  rec.onend = () => {
    if (mic.listening) stopMic();
  };
  mic.rec = rec;
  mic.listening = true;
  setMicUi(true);
  try {
    rec.start();
  } catch {
    stopMic();
  }
}

$("auth-form").addEventListener("submit", (event) => {
  event.preventDefault();
  submitAuth();
});
$("auth-register").addEventListener("click", () => {
  setAuthMode(!state.authRegister);
  if (state.authRegister) ($("auth-confirm").value ? $("auth-login") : $("auth-confirm")).focus();
});
document.querySelectorAll(".pass-toggle").forEach((button) => {
  button.addEventListener("click", () => setPasswordVisible(!state.showPassword));
});
$("logout").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST" });
  state.user = null;
  state.chat = null;
  setAuthMode(false);
  setPasswordVisible(false);
  $("auth-form").reset();
  show("app-view", false);
  show("login-view", true);
});
$("new-chat").addEventListener("click", createChat);
$("confirm-cancel").addEventListener("click", closeConfirm);
$("confirm-ok").addEventListener("click", confirmDeleteChat);
$("confirm-dialog").addEventListener("click", (event) => {
  if (event.target.id === "confirm-dialog") closeConfirm();
});
function setChatSearchOpen(on) {
  const open = Boolean(on);
  show("search-chats", !open);
  show("search-field", open);
  $("nav-search").classList.toggle("open", open);
  if (open) {
    $("chat-filter").focus();
    return;
  }
  $("chat-filter").value = "";
  renderChatList();
}

$("search-chats").addEventListener("click", () => setChatSearchOpen(true));
$("chat-filter").addEventListener("input", renderChatList);
$("chat-filter").addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    setChatSearchOpen(false);
  }
});
document.addEventListener("mousedown", (event) => {
  if ($("search-field").hidden) return;
  if ($("nav-search").contains(event.target)) return;
  setChatSearchOpen(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("confirm-dialog").hidden) closeConfirm();
  else if (event.key === "Escape" && !$("search-field").hidden) setChatSearchOpen(false);
});
$("open-pdfs").addEventListener("click", () => show("pdf-drawer", true));
$("close-pdfs").addEventListener("click", () => show("pdf-drawer", false));
$("upload-pdf").addEventListener("click", () => $("file-input").click());
$("attach-btn").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  event.target.value = "";
  await uploadPdf(file);
});
$("prompt").addEventListener("input", resizePrompt);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendPrompt();
  }
});
$("send-btn").addEventListener("click", () => {
  if (state.busy) cancelSend();
  else sendPrompt();
});
$("model-btn").addEventListener("click", (event) => {
  renderModelMenu();
  placeMenu($("model-menu"), event.currentTarget);
});
$("settings-btn").addEventListener("click", (event) => {
  const menu = $("settings-menu");
  for (const button of menu.querySelectorAll("button[data-route]")) {
    button.classList.toggle("active", button.dataset.route === state.routeMode);
  }
  placeMenu(menu, event.currentTarget);
});
$("settings-menu").addEventListener("click", (event) => {
  const route = event.target.dataset.route;
  if (!route) return;
  state.routeMode = route;
  $("settings-menu").hidden = true;
});
$("toggle-sidebar").addEventListener("click", () => {
  $("app-view").classList.toggle("sidebar-open");
});
$("mic-btn").addEventListener("click", toggleMic);
document.addEventListener("click", (event) => {
  if (!$("model-menu").contains(event.target) && event.target !== $("model-btn") && !event.target.closest("#model-btn")) {
    $("model-menu").hidden = true;
  }
  if (!$("settings-menu").contains(event.target) && event.target !== $("settings-btn") && !event.target.closest("#settings-btn")) {
    $("settings-menu").hidden = true;
  }
});

boot();
