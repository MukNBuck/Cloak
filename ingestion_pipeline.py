import os
import base64
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

def get_image_base64(image_path):
    """Convert an image to base64 so Ollama can read it."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def summarize_image_with_ollama(image_path):
    """Send the extracted image to your GTX 1060 vision model."""
    print(f"Sending {os.path.basename(image_path)} to qwen2.5vl:7b for visual parsing...")
    
    # Initialize your specific vision model
    vision_llm = ChatOllama(model="qwen2.5vl:7b", temperature=0.0)
    
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
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=output_image_dir,
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
                
                # Optional: Delete the image after processing to save space, or keep it to return in the UI
                # os.remove(img_path) 

    return all_documents
