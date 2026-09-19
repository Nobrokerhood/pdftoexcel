from app.workflows.state import AccountingWorkflowState


def route_after_verification(state: AccountingWorkflowState) -> str:
    if state.get("overall_status") == "FAILED":
        return "human_review"
    if state.get("verification_status") == "PASSED":
        return "map"
    # A repair that was discarded for dropping rows will be discarded again on the
    # next identical attempt, so retrying only burns Gemini calls and latency.
    # Escalate to human review instead.
    extracted = state.get("extracted_data") or {}
    if extracted.get("repair_status") == "REJECTED_ROW_LOSS":
        return "human_review"
    if state.get("extraction_attempt", 0) <= state.get("max_retries", 2):
        return "repair"
    return "human_review"


def route_after_mapping(state: AccountingWorkflowState) -> str:
    if state.get("mapping_status") == "NEEDS_MAPPING":
        return "human_review"
    return "validate"


def route_after_validation(state: AccountingWorkflowState) -> str:
    return "human_review"
