import os
from pathlib import Path

import requests

# CONFIGURATION: Use your laptop's network IP address
SERVER_IP = "192.168.1.146"
BASE_URL = f"http://{SERVER_IP}:8001"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

def upload_test_pdf(pdf_path):
    print(f"\n[1] Sending '{pdf_path}' to server over the network...")
    if not os.path.exists(pdf_path):
        print(f"Error: Please drop a sample PDF named '{pdf_path}' in this folder first!")
        return False
        
    url = f"{BASE_URL}/upload_pdf"
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(url, files=files)
        
    if response.status_code == 200:
        print("-> Success! Server response:")
        print(response.json())
        return True
    else:
        print(f"-> Failed! Status {response.status_code}: {response.text}")
        return False

def query_test_rag(question_text):
    print(f"\n[2] Sending RAG query to server: '{question_text}'...")
    url = f"{BASE_URL}/query"
    payload = {"query": question_text}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("\n================ ANSWER FROM SERVER ================")
        print(data.get("answer"))
        print("===========================================================")
        
        print("\n[3] Source document excerpts returned to browser:")
        for i, excerpt in enumerate(data.get("context", []), 1):
            print(f"\nExcerpt {i}:")
            print(excerpt[:250] + "..." if len(excerpt) > 250 else excerpt)
    else:
        print(f"-> Failed! Status {response.status_code}: {response.text}")

if __name__ == "__main__":
    target_pdf = str(SAMPLES_DIR / "test.pdf")
    
    # 1. Test uploading and indexing
    if upload_test_pdf(target_pdf):
        # 2. Test retrieving and calculating answers
        # (Change this question to match something inside your test.pdf)
        user_query = input()
        query_test_rag(user_query)