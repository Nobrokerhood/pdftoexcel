"""Transaction candidate lifecycle and row-meaningfulness guards.

Central rule enforced here: a row that carries no source-supported accounting
value is never a transaction. It is either an explicitly rejected candidate
(with a reason) or an unresolved candidate routed to human review -- but it is
never fabricated to satisfy a schema, and never reported as VERIFIED.

Lifecycle:
    source_candidate -> classified_candidate -> accepted_transaction
                                             -> unresolved_candidate (human review)
                                             -> rejected_candidate (with reason)
"""

from dataclasses import dataclass, field, asdict
from typing import Any

from app.accounting.templates import NBH_IMPORT_COLUMNS


CANDIDATE_SCHEMA_VERSION = 1

# Values that carry no accounting meaning regardless of which column they sit in.
PLACEHOLDER_VALUES = {
    "", "-", "--", "none", "null", "n/a", "na", "nan",
    "unknown", "undefined", "not found", "not available",
}

# Columns that, on their own, establish that a row refers to a real transaction.
IDENTITY_COLUMNS = (
    "Cheque/Ref No*",
    "Transaction Date*",
    "Comments",
    "Bill Head*",
)

AMOUNT_COLUMN = "Amount*"


class CandidateDecision:
    ACCEPTED = "ACCEPTED"
    UNRESOLVED = "UNRESOLVED"
    REJECTED = "REJECTED"


def is_placeholder(value: Any) -> bool:
    """True when the value carries no accounting meaning."""
    if value is None:
        return True
    return str(value).strip().lower() in PLACEHOLDER_VALUES


def meaningful_fields(row: dict[str, Any]) -> list[str]:
    """Names of the NBH columns in this row that hold a real source value."""
    if not isinstance(row, dict):
        return []
    return [col for col in NBH_IMPORT_COLUMNS if not is_placeholder(row.get(col))]


def row_is_meaningful(row: dict[str, Any]) -> bool:
    """True when the row holds at least one real accounting value.

    A row of nothing but '-' is not a transaction; it is an empty shell. This is
    the guard that stops an all-placeholder row being created or verified.
    """
    return bool(meaningful_fields(row))


def row_has_amount(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    return not is_placeholder(row.get(AMOUNT_COLUMN))


def row_has_identity(row: dict[str, Any]) -> bool:
    """True when the row identifies *which* transaction it is, amount aside."""
    if not isinstance(row, dict):
        return False
    return any(not is_placeholder(row.get(col)) for col in IDENTITY_COLUMNS)


def classify_row_completeness(row: dict[str, Any]) -> tuple[str, str]:
    """Return (decision, reason) for a prepared NBH row.

    An amount alone, or an identity alone, is enough to keep the row as a
    candidate -- losing a real transaction is worse than carrying an uncertain
    one into review. Only a row with neither is rejected.
    """
    if not isinstance(row, dict):
        return CandidateDecision.REJECTED, "Row is not a transaction object."

    has_amount = row_has_amount(row)
    has_identity = row_has_identity(row)

    if has_amount and has_identity:
        return CandidateDecision.ACCEPTED, ""
    if has_amount and not has_identity:
        return (
            CandidateDecision.UNRESOLVED,
            "Amount found in source but no reference, date or description could be "
            "read to identify the transaction.",
        )
    if has_identity and not has_amount:
        return (
            CandidateDecision.UNRESOLVED,
            "Transaction identified in source but the amount could not be read "
            "confidently; preserved for human review rather than discarded.",
        )
    return (
        CandidateDecision.REJECTED,
        "No amount and no identifying field could be read from the source for this row.",
    )


@dataclass
class RejectedCandidate:
    """A candidate deliberately NOT exported, with the reason recorded.

    Every discarded candidate produces one of these so row loss is always
    provable rather than silent.
    """

    candidate_index: int
    source_text: str = ""
    classification: str = "UNKNOWN"
    decision: str = CandidateDecision.REJECTED
    reason: str = ""
    page: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    amount_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateLedger:
    """Accounting of every candidate seen, so totals always reconcile.

    Invariant: source_candidates == accepted + unresolved + len(rejected)
    """

    schema_version: int = CANDIDATE_SCHEMA_VERSION
    source_candidates: int = 0
    accepted: int = 0
    unresolved: int = 0
    rejected: list[RejectedCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def reject(self, candidate: RejectedCandidate) -> None:
        self.rejected.append(candidate)

    @property
    def balanced(self) -> bool:
        return self.source_candidates == self.accepted + self.unresolved + len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_candidates": self.source_candidates,
            "accepted": self.accepted,
            "unresolved": self.unresolved,
            "rejected_count": len(self.rejected),
            "rejected": [item.to_dict() for item in self.rejected],
            "balanced": self.balanced,
            "notes": list(self.notes),
        }
