from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.core.errors import ServiceNotConfiguredError, service_unavailable
from app.documents.conversion import convert_to_template_csv, export_as_excel
from app.documents.pdf_splitter import split_pdf_to_zip


router = APIRouter()


def _services(request: Request):
    return request.app.state.settings, request.app.state.gemini_client


@router.post("/process-document/")
async def process_document(request: Request, file: UploadFile = File(...)):
    settings, gemini_client = _services(request)
    try:
        csv_text = await convert_to_template_csv(
            file, settings.max_file_size_mb, gemini_client
        )
    except ServiceNotConfiguredError as exc:
        raise service_unavailable(str(exc)) from exc

    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=converted_template.csv"},
    )


@router.post("/export-to-excel/")
async def export_to_excel(request: Request, file: UploadFile = File(...)):
    settings, gemini_client = _services(request)
    try:
        excel_buf = await export_as_excel(file, settings.max_file_size_mb, gemini_client)
    except ServiceNotConfiguredError as exc:
        raise service_unavailable(str(exc)) from exc

    return StreamingResponse(
        excel_buf,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={"Content-Disposition": "attachment; filename=exported_data.xlsx"},
    )


@router.post("/split-pdf/")
async def split_pdf(
    request: Request,
    file: UploadFile = File(...),
    pages_per_file: int = 5,
):
    settings = request.app.state.settings
    zip_buf = await split_pdf_to_zip(file, settings.max_file_size_mb, pages_per_file)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=split_parts.zip"},
    )
