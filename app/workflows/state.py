from typing import Any, TypedDict


class AccountingWorkflowState(TypedDict, total=False):
    workflow_id: str
    job_id: str
    session_id: str
    user_email: str
    source_filename: str
    source_drive_file_id: str
    source_type: str
    selected_purpose: str
    template_code: str
    input_folder_id: str
    review_folder_id: str
    completed_folder_id: str
    output_folder_id: str
    extraction_attempt: int
    extracted_data: dict[str, Any]
    verification_result: dict[str, Any]
    verification_status: str
    mapping_result: dict[str, Any]
    mapping_status: str
    validation_result: dict[str, Any]
    validation_status: str
    human_corrections: list[dict[str, Any]]
    human_status: str
    output_filename: str
    output_drive_file_id: str
    current_step: str
    overall_status: str
    last_error: str
