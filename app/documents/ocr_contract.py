"""The single OCR contract every provider implements.

Consumers never instantiate an engine: they receive an `OcrOrchestrator`, which
is the only code that selects providers. Every provider returns an `OcrResult`
so results from different engines and preprocessing variants are directly
comparable and can all be retained as evidence.

Coordinates are always pixels of the page image the provider was given. The
orchestrator only ever gives providers the page's canonical image (or a
same-size pixel filter of it), so every result on a page shares one coordinate
space, and that same image is what Gemini is shown.
"""

from dataclasses import asdict, dataclass, field
from typing import Protocol

from PIL import Image

GEOMETRY_REAL = "REAL"            # measured from pixels or PDF text positions
GEOMETRY_ESTIMATED = "ESTIMATED"  # synthesised; must not be used for alignment


@dataclass(frozen=True)
class OcrLineEvidence:
    line_id: str
    engine: str
    variant: str
    page: int
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in canonical page pixels

    @property
    def x0(self) -> int:
        return self.bbox[0]

    @property
    def x1(self) -> int:
        return self.bbox[2]

    @property
    def y0(self) -> int:
        return self.bbox[1]

    @property
    def y1(self) -> int:
        return self.bbox[3]

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "OcrLineEvidence":
        return cls(
            line_id=str(data["line_id"]),
            engine=str(data.get("engine", "")),
            variant=str(data.get("variant", "")),
            page=int(data.get("page", 1)),
            text=str(data.get("text", "")),
            confidence=float(data.get("confidence", 0.0)),
            bbox=tuple(int(v) for v in data.get("bbox", (0, 0, 0, 0))),
        )


@dataclass
class OcrResult:
    engine: str
    page: int
    variant: str
    width: int
    height: int
    lines: list[OcrLineEvidence] = field(default_factory=list)
    duration_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    geometry: str = GEOMETRY_REAL

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return round(sum(line.confidence for line in self.lines) / len(self.lines), 4)

    @property
    def key(self) -> str:
        return f"{self.engine}:{self.variant}"

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "page": self.page,
            "variant": self.variant,
            "width": self.width,
            "height": self.height,
            "duration_ms": self.duration_ms,
            "warnings": list(self.warnings),
            "geometry": self.geometry,
            "lines": [line.to_dict() for line in self.lines],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OcrResult":
        return cls(
            engine=str(data.get("engine", "")),
            page=int(data.get("page", 1)),
            variant=str(data.get("variant", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            lines=[OcrLineEvidence.from_dict(item) for item in data.get("lines", [])],
            duration_ms=int(data.get("duration_ms", 0)),
            warnings=list(data.get("warnings", [])),
            geometry=str(data.get("geometry", GEOMETRY_REAL)),
        )


def make_line_id(engine: str, variant: str, page: int, index: int) -> str:
    return f"p{page}.{engine}.{variant}.{index}"


class OcrProvider(Protocol):
    """Implemented by every OCR engine. Only the orchestrator calls it."""

    name: str

    def available(self) -> bool:
        ...

    @property
    def unavailable_reason(self) -> str:
        ...

    def read(self, image: Image.Image, page_idx: int | None = None, variant: str | None = None) -> OcrResult:
        ...
