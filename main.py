import os
import io
import json
import time
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pdf2image import convert_from_bytes
from PIL import Image
from dotenv import load_dotenv

# --- Load .env variables ---
load_dotenv()

# --- Configuration & Startup Check ---
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ GEMINI_API_KEY loaded successfully.")
except KeyError:
    raise RuntimeError("FATAL: GEMINI_API_KEY environment variable not set.")

app = FastAPI()

# --- Allow all origins (you can restrict later for security) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Gemini model ---
model = genai.GenerativeModel("gemini-2.5-flash")


# ------------------- PROMPTS -------------------
def create_discovery_prompt():
    """Creates a prompt to discover the column headers from the document."""
    return """
    Analyze the provided image of a ledger or bill.
    Identify all the unique charge or expense column headers in the table.
    Return the result as a clean, comma-separated list.
    Example: Property Tax,Water Charges,Sinking Fund,Maint. Charges
    """


def create_extraction_prompt(discovered_headers):
    """Creates a dynamic prompt for structured data extraction."""
    return f"""
    You are an expert data entry clerk. Analyze the provided image of a ledger.
    Based on the following charge categories: {', '.join(discovered_headers)}
    Extract data for every member into a valid JSON array.

    RULES:
    1. For each member, extract Wing, Unit No, and Member Name.
    2. If a name is missing, use null.
    3. For each charge, extract the numeric value or null if blank.
    4. Output only the raw JSON array.

    JSON Schema Example:
    [
        {{
            "Wing": "string or null",
            "Unit No": "string or null",
            "Member Name": "string or null",
            "Charges": {{
                "{discovered_headers[0]}": "float or null",
                "{discovered_headers[1]}": "float or null"
            }}
        }}
    ]
    """


def create_direct_export_prompt():
    """Prompt for direct table-to-JSON export."""
    return """
    You are an expert at table data extraction.
    Analyze the provided image and extract the main table as JSON array.
    Each row should be one JSON object using the headers as keys.
    Do not omit or transform data.
    Output only raw JSON.
    """


# ------------------- Utility: Retry Helper -------------------
def safe_generate(prompt_list, retries=2):
    """Retry Gemini model calls for reliability."""
    for attempt in range(retries):
        try:
            return model.generate_content(prompt_list)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)


# ------------------- Utility: File/Image Handling -------------------
async def get_image_from_upload(file: UploadFile):
    """Handles both images and multi-page PDFs."""
    if not file.content_type in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    file_bytes = await file.read()

    if file.content_type == "application/pdf":
        try:
            images = convert_from_bytes(file_bytes)
            if not images:
                raise HTTPException(status_code=400, detail="Could not extract images from PDF.")
            return images  # return list of pages
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
    else:
        return [Image.open(io.BytesIO(file_bytes))]  # return single image as list


# ------------------- ENDPOINT 1: Process to CSV Template -------------------
@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    images_to_process = await get_image_from_upload(file)
    all_rows = []

    # Step 1: Discover headers from first page only
    try:
        discovery_prompt = create_discovery_prompt()
        response = safe_generate([discovery_prompt, images_to_process[0]])
        header_text = response.text.strip()
        discovered_headers = [h.strip() for h in header_text.split(',') if h.strip()]
        if not discovered_headers:
            raise HTTPException(status_code=400, detail="No expense headers identified.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Header discovery failed: {e}")

    # Step 2: Extract data from ALL pages
    for page_index, image_page in enumerate(images_to_process, start=1):
        try:
            extraction_prompt = create_extraction_prompt(discovered_headers)
            response = safe_generate([extraction_prompt, image_page])
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            extracted_data = json.loads(json_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Extraction failed on page {page_index}: {e}")

        # Flatten extracted data
        for member in extracted_data:
            flat_row = {
                "Bill Number": f"{member.get('Wing', '') or ''}-{member.get('Unit No', '') or ''}".strip('-'),
                "Bill Date": None,
                "Vendor Code": None,
                "Due Date": None,
                "Narration": member.get("Member Name"),
                "CGST Tax Ledger Code": None, "CGST Amount": None,
                "SGST Tax Ledger Code": None, "SGST Amount": None,
                "IGST Tax Ledger Code": None, "IGST Amount": None,
                "TDS Code": None, "TDS Amount": None
            }

            charges = member.get("Charges", {})
            for i, header in enumerate(discovered_headers):
                column_number = i + 1
                if column_number > 10:
                    break
                amount = charges.get(header)
                flat_row[f"Expense Code {column_number}"] = header
                flat_row[f"Expense Amount {column_number}"] = amount

            # Fill remaining columns
            for i in range(len(discovered_headers), 10):
                column_number = i + 1
                flat_row[f"Expense Code {column_number}"] = None
                flat_row[f"Expense Amount {column_number}"] = None

            all_rows.append(flat_row)

    if not all_rows:
        raise HTTPException(status_code=400, detail="No data could be processed.")

    df = pd.DataFrame(all_rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=converted_template.csv"}
    )


# ------------------- ENDPOINT 2: Direct Export to Excel -------------------
@app.post("/export-to-excel/")
async def export_to_excel(file: UploadFile = File(...)):
    images_to_process = await get_image_from_upload(file)
    all_data = []

    try:
        direct_export_prompt = create_direct_export_prompt()
        for page_index, image_page in enumerate(images_to_process, start=1):
            response = safe_generate([direct_export_prompt, image_page])
            json_text = response.text.strip().replace("```json", "").replace("```", "")
            extracted_data = json.loads(json_text)
            all_data.extend(extracted_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data extraction failed: {e}")

    if not all_data:
        raise HTTPException(status_code=400, detail="No data extracted for Excel export.")

    df = pd.DataFrame(all_data)
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name="Extracted Data", engine="openpyxl")
    excel_buffer.seek(0)

    return StreamingResponse(
        excel_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=exported_data.xlsx"}
    )
