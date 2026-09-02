import gc
import io
from zipfile import ZipFile

from fastapi import HTTPException, UploadFile
from PyPDF2 import PdfReader, PdfWriter


async def split_pdf_to_zip(
    file: UploadFile,
    max_file_size_mb: int,
    pages_per_file: int = 5,
) -> io.BytesIO:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if pages_per_file < 1:
        raise HTTPException(
            status_code=400, detail="pages_per_file must be greater than zero."
        )

    file_bytes = await file.read()
    if len(file_bytes) > max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"File exceeds {max_file_size_mb} MB limit."
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        total = len(reader.pages)
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file.")

        parts = []
        for start in range(0, total, pages_per_file):
            writer = PdfWriter()
            for page_index in range(start, min(start + pages_per_file, total)):
                writer.add_page(reader.pages[page_index])
            part_io = io.BytesIO()
            writer.write(part_io)
            part_io.seek(0)
            parts.append((f"part_{start // pages_per_file + 1}.pdf", part_io))
            gc.collect()

        zip_buf = io.BytesIO()
        with ZipFile(zip_buf, "w") as zip_file:
            for name, part_data in parts:
                zip_file.writestr(name, part_data.read())
        zip_buf.seek(0)
        return zip_buf
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"PDF splitting failed: {exc}"
        ) from exc
