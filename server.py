import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ingestion_pipeline import load_documents, split_documents, create_vector_store
from history_aware_generation import ask_question

app = FastAPI(title="Local RAG Backend API")

os.makedirs("docs", exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    file_path = os.path.join("docs", file.filename)
    
    # 1. Save the uploaded file to the laptop
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"\n--- NEW FILE UPLOADED: {file.filename} ---")
    
    # 2. Trigger your ingestion pipeline
    try:
        documents = load_documents(docs_path="docs")
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks)
        
        return {
            "status": "success", 
            "message": f"{file.filename} ingested successfully.",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    print(f"\n--- NEW RAG QUERY RECEIVED ---")
    print(f"Query: {request.query}")
    
    # 1. Run the history-aware RAG pipeline
    result = ask_question(request.query)
    
    # 2. Return the parsed answer and context excerpts back to the client
    return result
