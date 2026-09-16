"""Field values with source evidence and confidence."""

from dataclasses import asdict, dataclass, field
from decimal import Decimal

from app.documents.ocr import OcrLine


FOUND = "FOUND"
MISSING = "MISSING"
AMBIGUOUS = "AMBIGUOUS"

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


@dataclass
class Evidence:
    page: int | None
    text: str
    bbox: list[int] | None = None
    ocr_confidence: float | None = None
    engine: str = ""
    label: str = ""
    row: int | None = None

    @classmethod
    def from_line(cls, line: OcrLine, label: str = "", row: int | None = None) -> "Evidence":
        return cls(
            page=line.page,
            text=line.text,
            bbox=list(line.bbox),
            ocr_confidence=line.confidence,
            engine=line.engine,
            label=label,
            row=row,
        )

    def describe(self) -> str:
        where = f"page {self.page}" if self.page else "source"
        if self.row is not None:
            where += f", row {self.row}"
        prefix = f"{self.label}: " if self.label else ""
        return f"{where}: \"{prefix}{self.text}\""


def _jsonable(value):
    if isinstance(value, Decimal):
        return format(value.normalize(), "f") if value == value.to_integral() else str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@dataclass
class FieldResult:
    field: str
    value: object = None
    status: str = MISSING
    confidence: str = LOW
    score: float = 0.0
    evidence: Evidence | None = None
    candidates: list = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    repaired: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["value"] = _jsonable(self.value)
        data["candidates"] = [_jsonable(c) for c in self.candidates]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FieldResult":
        evidence = data.get("evidence")
        return cls(
            field=data["field"],
            value=data.get("value"),
            status=data.get("status", MISSING),
            confidence=data.get("confidence", LOW),
            score=float(data.get("score", 0.0)),
            evidence=Evidence(**evidence) if evidence else None,
            candidates=list(data.get("candidates", [])),
            reasons=list(data.get("reasons", [])),
            repaired=bool(data.get("repaired", False)),
        )


def missing(field_name: str, reason: str) -> FieldResult:
    return FieldResult(field=field_name, status=MISSING, confidence=LOW, score=0.0, reasons=[reason])
