import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
# UPDATED: Importing the vision-specific function
from ingestion_pipeline import load_documents_with_vision, split_documents, create_vector_store, get_existing_vectorstore
from history_aware_generation import ask_question

app = FastAPI(title="Local RAG Backend API")

os.makedirs("docs", exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    
    # 1. Get the DB connection cleanly from your pipeline
    vectorstore = get_existing_vectorstore()

    # 2. Check if the file already exists
    existing_data = vectorstore.get(where={"source": file.filename})

    if len(existing_data.get("ids", [])) > 0:
        print(f"Skipping {file.filename}: Already exists in database.")
        return {"status": "skipped", "message": f"'{file.filename}' is already embedded."}

    # 3. Proceed with normal upload
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join("docs", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"\n--- NEW FILE UPLOADED: {file.filename} ---")

    try:
        documents = load_documents_with_vision(docs_path="docs")
        chunks = split_documents(documents)
        create_vector_store(chunks)

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
    
    result = ask_question(request.query)
    return result
