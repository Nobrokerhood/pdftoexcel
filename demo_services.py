"""In-memory sample services for the standalone local demo.
Adapted from the repository's workflow test fixtures; no live credentials.
"""
import copy
from app.core.config import Settings
from app.auth.google_auth import AuthError
def settings(max_retries=2) -> Settings:
    return Settings(
        google_client_id="client-id",
        allowed_email_domain="nobroker.in",
        allow_domain_wide_access=False,
        session_inactivity_seconds=1200,
        session_heartbeat_grace_seconds=120,
        ai_verification_max_retries=max_retries,
        allow_dev_login=False,
        gemini_api_key=None,
        gemini_model="gemini-2.5-flash",
        max_file_size_mb=10,
        google_service_account_json=None,
        google_service_account_file=None,
        google_accounting_spreadsheet_id=None,
        google_user_master_sheet_id="users",
        google_login_audit_sheet_id="login",
        google_api_usage_sheet_id="usage",
        google_session_log_sheet_id="sessions",
        google_activity_log_sheet_id="activity",
        google_processing_log_sheet_id="processing",
        google_template_master_sheet_id="templates",
        google_folder_config_sheet_id="folders",
        google_mapping_master_sheet_id="mapping",
        google_drive_root_folder_id="root",
        google_login_audit_sheet_name="Accounting_AI_Login_Audit",
        google_api_usage_sheet_name="API_Usage_Report",
        cors_allowed_origins=(
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8031",
            "http://127.0.0.1:8031",
        ),
    )


class FakeSheetsService:
    def __init__(self, records):
        self.records = records
        self.appended = []
        self.updated = []

    def read_records(self, spreadsheet_id, worksheet_name="Sheet1", use_cache=True):
        return [row.copy() for row in self.records.get(spreadsheet_id, [])]

    def read_table(self, table_key, use_cache=True):
        table_map = {
            "user_master": "users",
            "folder_config": "folders",
            "template_master": "templates",
            "mapping_master": "mapping",
            "login_audit": "login",
            "activity_log": "activity",
            "session_log": "sessions",
            "processing_log": "processing",
            "job_state": "job_state",
        }
        return self.read_records(table_map.get(table_key, table_key))

    def append_row(self, spreadsheet_id, values, worksheet_name="Sheet1"):
        self.appended.append((spreadsheet_id, values))
        return True

    def append_table_row(self, table_key, values):
        table_map = {
            "login_audit": "login",
            "activity_log": "activity",
            "session_log": "sessions",
            "processing_log": "processing",
            "job_state": "job_state",
        }
        return self.append_row(table_map.get(table_key, table_key), values)

    def lookup_row_by_key(self, spreadsheet_id, key_column, key_value, worksheet_name="Sheet1"):
        for record in self.read_records(spreadsheet_id, worksheet_name):
            if str(record.get(key_column, "")).strip().lower() == key_value.lower():
                return record
        return None

    def lookup_table_row_by_key(self, table_key, key_column, key_value):
        for record in self.read_table(table_key):
            if str(record.get(key_column, "")).strip().lower() == key_value.lower():
                return record
        return None

    def update_row_by_key(self, spreadsheet_id, key_column, key_value, updates, worksheet_name="Sheet1"):
        self.updated.append((spreadsheet_id, key_column, key_value, updates))
        return True

    def update_table_row_by_key(self, table_key, key_column, key_value, updates):
        table_map = {
            "login_audit": "login",
            "processing_log": "processing",
            "job_state": "job_state",
        }
        return self.update_row_by_key(table_map.get(table_key, table_key), key_column, key_value, updates)


class FakeVerifier:
    def verify(self, credential):
        raise AuthError("Google sign-in is disabled in the local demo. Use the demo entry page.")


class FakeDriveService:
    def __init__(self):
        self.uploads = []
        self.moves = []
        self.downloads = {}

    def upload_file(self, filename, content, folder_id, mime_type="application/octet-stream"):
        file_id = f"{folder_id}-{len(self.uploads) + 1}"
        if isinstance(content, bytes):
            stored_content = content
        else:
            stored_content = content.read()
        self.uploads.append(
            {
                "filename": filename,
                "content": stored_content,
                "folder_id": folder_id,
                "mime_type": mime_type,
                "file_id": file_id,
            }
        )
        self.downloads[file_id] = stored_content
        return file_id

    def move_file(self, file_id, folder_id):
        self.moves.append((file_id, folder_id))
        return True

    def download_file(self, file_id):
        return self.downloads[file_id]


class StaticExtractionProvider:
    def __init__(self, data):
        self.data = data

    def extract(self, source_bytes, purpose, template):
        return copy.deepcopy(self.data[purpose])


class StaticVerificationProvider:
    def __init__(self, results=None):
        self.results = list(results or [{"overall_status": "PASSED", "fields": []}])
        self.calls = 0

    def verify(self, source_bytes, purpose, template, extracted_data):
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[index]


class StaticRepairProvider:
    def __init__(self, repaired):
        self.repaired = repaired
        self.calls = 0

    def repair(self, source_bytes, purpose, template, extracted_data, verification_result):
        self.calls += 1
        return self.repaired.copy()


def records(include_mapping=True):
    mapping = []
    if include_mapping:
        mapping = [
            {"Purpose": "MEMBER_RECEIPT", "Type": "BANK", "Source Value": "HDFC", "Target Value": "HDFC001", "Active": "true"},
            {"Purpose": "MEMBER_RECEIPT", "Type": "BILL_HEAD", "Source Value": "Maintenance", "Target Value": "MAINT", "Active": "true"},
            {"Purpose": "VENDOR_INVOICE", "Type": "VENDOR", "Source Value": "ABC Plumbing Services Pvt Ltd", "Target Value": "VEND-ABC", "Active": "true"},
            {"Purpose": "VENDOR_INVOICE", "Type": "EXPENSE", "Source Value": "Plumbing work", "Target Value": "REPAIR", "Active": "true"},
        ]
    return {
        "users": [
            {"Email": "demo@example.test", "Name": "Demo User", "Role": "USER", "Active": "true"}
        ],
        "folders": [
            {"Purpose": "MEMBER_RECEIPT", "Incoming Folder ID": "member-in", "Review Folder ID": "member-review", "Completed Folder ID": "member-done", "Output Folder ID": "member-out", "Active": "true"},
            {"Purpose": "VENDOR_INVOICE", "Incoming Folder ID": "vendor-in", "Review Folder ID": "vendor-review", "Completed Folder ID": "vendor-done", "Output Folder ID": "vendor-out", "Active": "true"},
        ],
        "mapping": mapping,
    }


MEMBER_DATA = {
    "payment_type": "UPI",
    "bank_name_or_code": "HDFC",
    "reference_number": "UPI123456",
    "tower": "A",
    "flat": "101",
    "bill_head": "Maintenance",
    "amount": "5000",
    "transaction_date": "25-Aug-2026",
    "comments": "SYNTHETIC TEST DATA",
    "meter_number": None,
    "cheque_issuer_bank": None,
    "cheque_date": None,
}

VENDOR_DATA = {
    "bill_number": "INV-418",
    "bill_date": "25-Aug-2026",
    "vendor_code": None,
    "vendor_name": "ABC Plumbing Services Pvt Ltd",
    "due_date": None,
    "narration": "SYNTHETIC TEST DATA",
    "cgst_amount": "900",
    "sgst_amount": "900",
    "igst_amount": "0",
    "tds_amount": "0",
    "expenses": [
        {"expense_code": None, "expense_description": "Plumbing work", "expense_amount": "10000"}
    ],
}


