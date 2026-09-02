from app.workflows.state import AccountingWorkflowState


def route_after_verification(state: AccountingWorkflowState) -> str:
    if state.get("overall_status") == "FAILED":
        return "human_review"
    if state.get("verification_status") == "PASSED":
        return "map"
    if state.get("extraction_attempt", 0) <= state.get("max_retries", 2):
        return "repair"
    return "human_review"


def route_after_mapping(state: AccountingWorkflowState) -> str:
    if state.get("mapping_status") == "NEEDS_MAPPING":
        return "human_review"
    return "validate"


def route_after_validation(state: AccountingWorkflowState) -> str:
    return "human_review"
