from dataclasses import dataclass


@dataclass(frozen=True)
class SheetSchema:
    key: str
    tab_name: str
    legacy_setting_name: str | None
    headers: tuple[str, ...]


SHEET_SCHEMAS = {
    "user_master": SheetSchema(
        "user_master",
        "User_Master",
        "google_user_master_sheet_id",
        ("Email", "Name", "Role", "Active", "Created At", "Updated At"),
    ),
    "login_audit": SheetSchema(
        "login_audit",
        "Login_Audit",
        "google_login_audit_sheet_id",
        (
            "Session ID",
            "Email",
            "Name",
            "Login Time",
            "Logout Time",
            "Login Status",
            "IP",
            "User Agent",
        ),
    ),
    "session_log": SheetSchema(
        "session_log",
        "Session_Log",
        "google_session_log_sheet_id",
        (
            "Session ID",
            "Email",
            "Login At",
            "Last Seen At",
            "Logout At",
            "Session Duration Seconds",
            "Active Duration Seconds",
            "Status",
        ),
    ),
    "activity_log": SheetSchema(
        "activity_log",
        "Activity_Log",
        "google_activity_log_sheet_id",
        (
            "Timestamp",
            "Session ID",
            "User Email",
            "Job ID",
            "Action",
            "Purpose",
            "Source File ID",
            "Output File ID",
            "Status",
            "Details",
        ),
    ),
    "processing_log": SheetSchema(
        "processing_log",
        "Processing_Log",
        "google_processing_log_sheet_id",
        (
            "Job ID",
            "Session ID",
            "User Email",
            "Purpose",
            "Template Code",
            "Source Filename",
            "Source Drive File ID",
            "Source Folder ID",
            "Extraction Status",
            "Verification Status",
            "Mapping Status",
            "Validation Status",
            "Human Status",
            "Output Filename",
            "Output Drive File ID",
            "Overall Status",
            "Started At",
            "Completed At",
        ),
    ),
    "template_master": SheetSchema(
        "template_master",
        "Template_Master",
        "google_template_master_sheet_id",
        ("Purpose", "Template Code", "Template Name", "Version", "Output Format", "Active"),
    ),
    "folder_config": SheetSchema(
        "folder_config",
        "Folder_Config",
        "google_folder_config_sheet_id",
        (
            "Purpose",
            "Incoming Folder ID",
            "Review Folder ID",
            "Completed Folder ID",
            "Output Folder ID",
            "Active",
        ),
    ),
    "mapping_master": SheetSchema(
        "mapping_master",
        "Mapping_Master",
        "google_mapping_master_sheet_id",
        (
            "Mapping Type",
            "Purpose",
            "Society",
            "Source Value",
            "Normalized Source Value",
            "Canonical Code",
            "Canonical Name",
            "Alias",
            "Active",
            "Created By",
            "Created At",
            "Updated By",
            "Updated At",
        ),
    ),
    "job_state": SheetSchema(
        "job_state",
        "Job_State",
        None,
        ("Job ID", "Workflow ID", "Current Step", "Overall Status", "State JSON", "Updated At"),
    ),
}


def schema_for(key: str) -> SheetSchema:
    return SHEET_SCHEMAS[key]
