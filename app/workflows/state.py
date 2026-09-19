from typing import Any, TypedDict


class AccountingWorkflowState(TypedDict, total=False):
    job_id: str
    failed: bool
    repair_attempts: int
