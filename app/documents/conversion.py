import gc
import io
import json
import logging

import pandas as pd
from fastapi import HTTPException, UploadFile
from pdf2image import convert_from_bytes
from PIL import Image, UnidentifiedImageError

from app.core.errors import ServiceNotConfiguredError
from app.services.gemini_client import GeminiDocumentClient


logger = logging.getLogger(__name__)

SUPPORTED_DOCUMENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}


def create_template_prompt() -> str:
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


def create_direct_export_prompt(expected_columns: list[str] | None = None) -> str:
    columns_str = ""
    if expected_columns:
        columns_str = (
            f"EXPECTED COLUMNS (use these exact names and order): "
            f"{', '.join(expected_columns)}\n"
        )

    return f"""
    You are a meticulous financial auditor. Analyze the provided image of a table. Extract the data exactly as it appears.

    {columns_str}

    CRITICAL RULES:
    1. Extract ALL visible rows and columns from the table.
    2. For each row, carefully associate every value with its correct column header based on visual alignment.
    3. If a cell is visually empty or contains only a dash '-', you MUST use a null value.
    4. IMPORTANT: Always maintain consistent column order across all rows. Use the EXACT same column names and order from the first row for all subsequent rows.
    5. The final output must be a valid JSON array of row objects with consistent structure.
    6. Do not skip any rows - extract every single row visible in the table.
    7. Do not add or remove columns between rows - every row must have identical keys in identical order.
    8. Return ONLY the JSON array, no additional text.
    """


def _parse_json_array(text: str):
    cleaned = text.strip().replace("```json", "").replace("```", "")
    return json.loads(cleaned)


async def get_images_from_upload(file: UploadFile, max_file_size_mb: int):
    if file.content_type not in SUPPORTED_DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_bytes = await file.read()
    if len(file_bytes) > max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {max_file_size_mb} MB limit. Please split first.",
        )

    if file.content_type == "application/pdf":
        try:
            images = []
            for page in convert_from_bytes(file_bytes, dpi=100, fmt="jpeg"):
                images.append(page.convert("RGB"))
                page.close()
            gc.collect()
            return images
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"PDF processing failed: {exc}"
            ) from exc

    try:
        return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


async def convert_to_template_csv(
    file: UploadFile,
    max_file_size_mb: int,
    gemini_client: GeminiDocumentClient,
) -> str:
    images = await get_images_from_upload(file, max_file_size_mb)
    rows = []

    try:
        for img in images:
            try:
                resp = gemini_client.generate_content([create_template_prompt(), img])
                rows.extend(_parse_json_array(resp.text))
            except ServiceNotConfiguredError:
                raise
            except Exception as exc:
                logger.error("Template conversion page failed: %s", exc)
            finally:
                img.close()
                gc.collect()
    finally:
        gc.collect()

    if not rows:
        raise HTTPException(status_code=400, detail="No data could be processed.")

    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue()


async def export_as_excel(
    file: UploadFile,
    max_file_size_mb: int,
    gemini_client: GeminiDocumentClient,
) -> io.BytesIO:
    images = await get_images_from_upload(file, max_file_size_mb)
    all_data = []
    unified_columns = None

    try:
        for page_count, img in enumerate(images, start=1):
            try:
                prompt = create_direct_export_prompt(unified_columns)
                resp = gemini_client.generate_content([prompt, img])
                data = _parse_json_array(resp.text)
                logger.info("Page %s: Extracted %s rows", page_count, len(data))

                if unified_columns is None and data:
                    unified_columns = list(data[0].keys())
                    logger.info("Unified columns set: %s", unified_columns)

                if unified_columns:
                    all_data.extend(
                        {column: row.get(column, None) for column in unified_columns}
                        for row in data
                    )
                else:
                    all_data.extend(data)
            except ServiceNotConfiguredError:
                raise
            except Exception as exc:
                logger.error("Error processing page %s: %s", page_count, exc)
            finally:
                img.close()
                gc.collect()
    finally:
        gc.collect()

    if not all_data:
        raise HTTPException(status_code=400, detail="No data extracted for Excel.")

    df = pd.DataFrame(all_data).fillna("")
    excel_buf = io.BytesIO()
    df.to_excel(excel_buf, index=False, sheet_name="Extracted Data", engine="openpyxl")
    excel_buf.seek(0)
    return excel_buf
