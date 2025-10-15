
import os
import io
import json
import time
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pdf2image import convert_from_bytes
from PIL import Image
from dotenv import load_dotenv

# ------------------- Load Environment Variables -------------------
load_dotenv()

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ GEMINI_API_KEY loaded successfully.")
except KeyError:
    raise RuntimeError("❌ FATAL: GEMINI_API_KEY environment variable not set.")

# ------------------- Initialize FastAPI -------------------
app = FastAPI(title="NoBrokerHood PDF→Excel Converter")

# ✅ Explicitly allow your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nobrokerhood.github.io",
        "https://nobrokerhood.github.io/pdftoexcel",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Initialize Gemini Model -------------------
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------- Original PROMPTS -------------------
def create_template_prompt():
    return """
    You are an expert data entry clerk. Your task is to analyze the provided image of a ledger and convert it into a specific flattened CSV format.
    For each unique member, create a single JSON object. Extract all charge types and their amounts. Structure this information into a JSON array.

    RULES:
    1. The JSON output MUST be a clean, raw array and nothing else.
    2. For each member, create a unique "Bill Number" from their Wing and Unit No (e.g., "A-1").
    3. Place the member's name in the "Narration" field.
    4. Map each charge to the "Expense Code" and "Expense Amount" columns sequentially.
    5. If a value is missing, use null.

    JSON SCHEMA:
    [{"Bill Number": "string", "Bill Date": null, "Vendor Code": null, "Due Date": null, "Narration": "string", "CGST Tax Ledger Code": null, "CGST Amount": null, "SGST Tax Ledger Code": null, "SGST Amount": null, "IGST Tax Ledger Code": null, "IGST Amount": null, "TDS Code": null, "TDS Amount": null, "Expense Code 1": "string", "Expense Amount 1": "float", "... up to 10 expense pairs"}]
    """

def create_direct_export_prompt():
    return """
    You are a meticulous financial auditor. Analyze the provided image of a table. Extract the data exactly as it appears.
    CRITICAL RULES:
    1. For each row, carefully associate every value with its correct column header based on visual alignment.
    2. If a cell is visually empty or contains only a dash '-', you MUST use a `null` value. Do not invent data.
    3. The final output must be a valid JSON array of row objects. Do not include any other text.
    Example: [{"Header 1": "Value A", "Header 2": 123.45, "Header 3": null}]
    """

# ------------------- Retry Helper -------------------
def safe_generate(prompt_list, retries=2):
    for attempt in range(retries):
        try:
            return model.generate_content(prompt_list)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)

# ------------------- File/Image Handling -------------------
async def get_image_from_upload(file: UploadFile):
    if not file.content_type in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_bytes = await file.read()

    if file.content_type == "application/pdf":
        try:
            images = convert_from_bytes(file_bytes)
            if not images:
                raise HTTPException(status_code=400, detail="Could not extract images from PDF.")
            return images
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
    else:
        return [Image.open(io.BytesIO(file_bytes))]

# ------------------- CSV Template Endpoint -------------------
@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    images_to_process = await get_image_from_upload(file)
    all_rows = []

    for image in images_to_process:
        try:
            prompt = create_template_prompt()
            response = safe_generate([prompt, image])
            page_data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            all_rows.extend(page_data)
        except Exception as e:
            print(f"Error processing page for template: {e}")
            continue

    if not all_rows:
        raise HTTPException(status_code=400, detail="No data could be processed for the template.")

    df = pd.DataFrame(all_rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=converted_template.csv"}
    )

# ------------------- Excel Export Endpoint -------------------
@app.post("/export-to-excel/")
async def export_to_excel(file: UploadFile = File(...)):
    images_to_process = await get_image_from_upload(file)
    all_data = []

    for image in images_to_process:
        try:
            prompt = create_direct_export_prompt()
            response = safe_generate([prompt, image])
            page_data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
            all_data.extend(page_data)
        except Exception as e:
            print(f"Error processing page for excel: {e}")
            continue

    if not all_data:
        raise HTTPException(status_code=400, detail="No data could be extracted for Excel.")

    df = pd.DataFrame(all_data)
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name="Extracted Data", engine="openpyxl")
    excel_buffer.seek(0)

    return StreamingResponse(
        excel_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=exported_data.xlsx"}
    )

# ------------------- Root Health Route -------------------
@app.get("/")
def root():
    return {"message": "✅ NoBrokerHood PDF→Excel API is running successfully on Render."}
