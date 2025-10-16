import os
import io
import json
import time
import gc
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pdf2image import convert_from_bytes
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from zipfile import ZipFile
from dotenv import load_dotenv

# ------------------- Load .env -------------------
load_dotenv()
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ GEMINI_API_KEY loaded successfully.")
except KeyError:
    raise RuntimeError("❌ FATAL: GEMINI_API_KEY environment variable not set.")

# ------------------- FastAPI App -------------------
app = FastAPI(title="NoBrokerHood PDF→Excel & Split Tool")

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

# ------------------- Gemini Model -------------------
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------- Prompts -------------------
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
    [{"Bill Number": "string", "Bill Date": null, "Vendor Code": null, "Due Date": null, "Narration": "string", "CGST Tax Ledger Code": null, "CGST Amount": null, "SGST Tax Ledger Code": null, "SGST Amount": null, "IGST Tax Ledger Code": null, "IGST Amount": null, "TDS Code": null, "TDS Amount": null, "Expense Code 1": "string", "Expense Amount 1": "float"}]
    """

def create_direct_export_prompt():
    return """
    You are a meticulous financial auditor. Analyze the provided image of a table. Extract the data exactly as it appears.
    CRITICAL RULES:
    1. For each row, carefully associate every value with its correct column header based on visual alignment.
    2. If a cell is visually empty or contains only a dash '-', you MUST use a `null` value.
    3. The final output must be a valid JSON array of row objects.
    """

# ------------------- Safe Generate -------------------
def safe_generate(prompt_list, retries=2):
    for attempt in range(retries):
        try:
            return model.generate_content(prompt_list)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)

# ------------------- File Handling -------------------
MAX_FILE_SIZE_MB = 10

async def get_images_from_upload(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    file_bytes = await file.read()

    # ✅ File size limit (10 MB)
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit. Please split first.")

    if file.content_type == "application/pdf":
        try:
            images = []
            for page in convert_from_bytes(file_bytes, dpi=100, fmt="jpeg"):
                images.append(page.convert("RGB"))
                page.close()
            gc.collect()
            return images
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
    else:
        return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

# ------------------- CSV Endpoint -------------------
@app.post("/process-document/")
async def process_document(file: UploadFile = File(...)):
    images = await get_images_from_upload(file)
    rows = []

    for img in images:
        try:
            resp = safe_generate([create_template_prompt(), img])
            data = json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
            rows.extend(data)
        except Exception as e:
            print("Error:", e)
        finally:
            img.close()
            gc.collect()

    if not rows:
        raise HTTPException(status_code=400, detail="No data could be processed.")

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=converted_template.csv"}
    )

# ------------------- Excel Endpoint -------------------
@app.post("/export-to-excel/")
async def export_to_excel(file: UploadFile = File(...)):
    images = await get_images_from_upload(file)
    all_data = []

    for img in images:
        try:
            resp = safe_generate([create_direct_export_prompt(), img])
            data = json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
            all_data.extend(data)
        except Exception as e:
            print("Error:", e)
        finally:
            img.close()
            gc.collect()

    if not all_data:
        raise HTTPException(status_code=400, detail="No data extracted for Excel.")

    df = pd.DataFrame(all_data)
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, sheet_name="Extracted Data", engine="openpyxl")
    excel_buf.seek(0)
    return StreamingResponse(
        excel_buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=exported_data.xlsx"}
    )

# ------------------- PDF Splitter Endpoint -------------------
@app.post("/split-pdf/")
async def split_pdf(file: UploadFile = File(...), pages_per_file: int = 5):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        total = len(reader.pages)
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file.")

        parts = []
        for start in range(0, total, pages_per_file):
            writer = PdfWriter()
            for i in range(start, min(start + pages_per_file, total)):
                writer.add_page(reader.pages[i])
            part_io = io.BytesIO()
            writer.write(part_io)
            part_io.seek(0)
            parts.append((f"part_{start//pages_per_file + 1}.pdf", part_io))
            gc.collect()

        zip_buf = io.BytesIO()
        with ZipFile(zip_buf, "w") as z:
            for name, fdata in parts:
                z.writestr(name, fdata.read())
        zip_buf.seek(0)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=split_parts.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF splitting failed: {e}")

# ------------------- Health Route -------------------
@app.get("/")
def root():
    return {"message": "✅ NoBrokerHood PDF→Excel & Split API running on Render."}
