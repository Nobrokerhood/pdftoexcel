import os
import io
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pdf2image import convert_from_bytes
from PIL import Image

# --- Configuration & Startup Check ---
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    raise RuntimeError("FATAL: GEMINI_API_KEY environment variable not set.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Be specific about the allowed origins
    allow_origins=["https://nobrokerhood.github.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST"], # You can also make methods more specific
    allow_headers=["*"],
)
model = genai.GenerativeModel("gemini-2.5-flash")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running and ready"}

# --- PROMPTS ---

def create_template_prompt():
    """Creates a single, powerful prompt for template conversion."""
    return """
    You are an expert data entry clerk. Your task is to analyze the provided image of a ledger and convert it into a specific flattened CSV format.
    First, identify all unique members listed. For each member, create a single record.
    Then, extract all charge types (e.g., 'Property Tax', 'Water Charges') and their corresponding amounts for that member.
    Finally, structure this information into a JSON array, where each object is a member.

    RULES:
    1.  The JSON output MUST be a clean, raw array and nothing else.
    2.  For each member, create a unique "Bill Number" from their Wing and Unit No (e.g., "A-1").
    3.  Place the member's name in the "Narration" field.
    4.  Map each of the member's charges to the "Expense Code" and "Expense Amount" columns sequentially.
    5.  If a value is missing, use null.

    JSON SCHEMA:
    [
        {
            "Bill Number": "string",
            "Bill Date": null,
            "Vendor Code": null,
            "Due Date": null,
            "Narration": "string",
            "CGST Tax Ledger Code": null, "CGST Amount": null,
            "SGST Tax Ledger Code": null, "SGST Amount": null,
            "IGST Tax Ledger Code": null, "IGST Amount": null,
            "TDS Code": null, "TDS Amount": null,
            "Expense Code 1": "string", "Expense Amount 1": "float",
            "Expense Code 2": "string", "Expense Amount 2": "float",
            "... up to 10 expense pairs"
        }
    ]
    """

def create_direct_export_prompt():
    """Creates a more robust prompt for direct-to-Excel export with a clear example."""
    return """
    You are an expert at table data extraction. Analyze the provided image and identify the main table.
    Extract all the data from the table exactly as it appears, row by row.
    Return the result as a valid JSON array of objects, where each object represents a row.
    Use the table's headers as the keys for each object.
    The output must only be the raw JSON array.

    For example, if the table has columns "Name" and "Amount", the output should be:
    [
        {
            "Name": "John Doe",
            "Amount": 100
        },
        {
            "Name": "Jane Smith",
            "Amount": 200
        }
    ]
    """

# --- Reusable function for file handling ---
async def get_images_from_upload(file: UploadFile):
    if not file.content_type in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    file_bytes = await file.read()
    if file.content_type == "application/pdf":
        try:
            images = convert_from_bytes(file_bytes, dpi=150)
            if not images: raise HTTPException(status_code=400, detail="Could not extract images.")
            return images
        except Exception as e: raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
    else:
        return [Image.open(io.BytesIO(file_bytes))]

# --- Function to clean up Gemini's response ---
def clean_and_parse_json(response_text: str):
    clean_text = response_text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        correction_prompt = f"The following text is not valid JSON. Please fix it and return only the raw, valid JSON object without any additional text or explanations:\n\n{clean_text}"
        correction_response = model.generate_content(correction_prompt)
        corrected_text = correction_response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(corrected_text)

# --- Endpoints ---
@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    images_to_process = await get_images_from_upload(file)
    all_rows = []
    for image in images_to_process:
        try:
            prompt = create_template_prompt()
            response = model.generate_content([prompt, image])
            page_data = clean_and_parse_json(response.text)
            all_rows.extend(page_data)
        except Exception: continue
    if not all_rows: raise HTTPException(status_code=400, detail="No data could be processed for the template.")
    df = pd.DataFrame(all_rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return StreamingResponse(iter([csv_buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=converted_template.csv"})

@app.post("/download-excel/")
async def export_to_excel(file: UploadFile = File(...)):
    images_to_process = await get_images_from_upload(file)
    all_data = []
    for image in images_to_process:
        try:
            prompt = create_direct_export_prompt()
            response = model.generate_content([prompt, image])
            page_data = clean_and_parse_json(response.text)
            all_data.extend(page_data)
        except Exception: continue
    if not all_data: raise HTTPException(status_code=400, detail="No data could be extracted for Excel.")
    df = pd.DataFrame(all_data)
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name="Extracted Data")
    excel_buffer.seek(0)
    return StreamingResponse(excel_buffer, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=exported_data.xlsx"})
