import os
import base64
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def get_image_base64(image_path):
    """Convert an image to base64 so Ollama can read it."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def summarize_image_with_ollama(image_path):
    """Send the extracted image to your GTX 1060 vision model."""
    print(f"Sending {os.path.basename(image_path)} to qwen2.5vl:7b for visual parsing...")
    
    # Initialize your specific vision model
    vision_llm = ChatOllama(model="llava", temperature=0.0)
    
    image_b64 = get_image_base64(image_path)
    
    # Prompt the model to describe the engineering diagram/image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image, chart, or diagram in meticulous detail. Include any numbers, dimensions, or text visible in the image. This will be used in a searchable engineering database."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
            ]
        }
    ]
    
    response = vision_llm.invoke(messages)
    return response.content

def load_documents_with_vision(docs_path="docs", output_image_dir="extracted_images"):
    """Extract text AND images, summarize images, and combine into documents."""
    print(f"Loading and partitioning PDFs from {docs_path}...")
    os.makedirs(docs_path, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    
    all_documents = []
    
    for filename in os.listdir(docs_path):
        if not filename.endswith(".pdf"):
            continue
            
        file_path = os.path.join(docs_path, filename)
        print(f"\nProcessing {filename}...")
        
        # 1. Partition the PDF (Extract text blocks and save images)
        raw_pdf_elements = partition_pdf(
            filename=file_path,
            strategy="fast",                     # Bypasses the missing engine crash
            extract_images_in_pdf=True,          # Keeps text parsing incredibly swift
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )
        
        # 2. Process Text Chunks
        for element in raw_pdf_elements:
            if "CompositeElement" in str(type(element)):
                doc = Document(
                    page_content=element.text,
                    metadata={"source": filename, "type": "text"}
                )
                all_documents.append(doc)
                
        # 3. Process Extracted Images using the Vision Model
        for image_name in os.listdir(output_image_dir):
            if image_name.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(output_image_dir, image_name)
                
                # Get the AI description of the image
                image_summary = summarize_image_with_ollama(img_path)
                
                # Store the AI description as searchable text, keeping the image path as metadata
                img_doc = Document(
                    page_content=f"Image Description: {image_summary}",
                    metadata={"source": filename, "type": "image", "image_path": img_path}
                )
                all_documents.append(img_doc)

    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]):
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:\n{chunk.page_content[:200]}...\n" + "-" * 50)

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="rag_collection",
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Finished creating ChromaDB vector store. Persisted at: {persist_directory}")

    return vectorstore

def get_existing_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    # Explicitly initialize the embedding model inside this function
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    return Chroma(
        persist_directory="db/chroma_db", 
        embedding_function=embedding_model,
        collection_name="rag_collection"
    )
