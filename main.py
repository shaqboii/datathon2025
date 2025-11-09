import os
import requests
from pathlib import Path
import mimetypes
import base64
from credentials import TOKEN

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
}

DATA_DIR = Path("../testCases")  # Your folder

def detect_file_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        return "unknown"
    if mime_type.startswith("image"):
        return "image"
    elif mime_type.startswith("text"):
        return "text"
    elif mime_type.startswith("application/pdf"):
        return "pdf"
    return "unknown"

def read_pdf_text(file_path):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed.")
        return ""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def file_to_message(file_path, file_type):
    if file_type == "image":
        # If endpoint accepts base64, send base64; if url, upload or provide path as per endpoint docs
        with open(file_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')
        # Format depends on endpoint: sometimes send as text, sometimes as base64 string or URL
        # Example (check endpoint docs!):
        return {"type": "text", "text": f"[Image file: {file_path.name}, encoded in base64]\n{encoded_img}"}
    elif file_type == "text":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()
        return {"type": "text", "text": text_content}
    elif file_type == "pdf":
        text_content = read_pdf_text(file_path)
        return {"type": "text", "text": text_content}
    else:
        return None

# Classification prompt
prompt_text = "Classify this content as one of: sensitive, classified, public, or unsafe. Additionally, detect any PII and state which page it is on. Additionally, describe why you classified a document as so, and what the PII violation was."

def query_cloud(messages, model="moonshotai/Kimi-K2-Thinking:novita"):
    payload = {
        "messages": messages,
        "model": model
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    try:
        return response.json()
    except Exception as e:
        print(f"Non-JSON or empty response: {response.status_code} {response.text}")
        return {"error": "Non-JSON or empty response", "status_code": response.status_code, "raw": response.text}

for file_path in DATA_DIR.iterdir():
    if file_path.is_file():
        file_type = detect_file_type(file_path)
        file_message = file_to_message(file_path, file_type)
        if not file_message:
            print(f"Skipping unknown file type: {file_path.name}")
            continue

        # Compose chat messages
        messages = [
            {"role": "user", "content": [file_message, {"type": "text", "text": prompt_text}]}
        ]

        # Query cloud model
        try:
            response = query_cloud(messages)
            print(f"File: {file_path.name}\nClassification: {response}\n")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
