"""Source candidate construction, BEFORE any accounting filtering.

Every visual row of every page becomes a `SourceCandidate` built directly from
OCR evidence. This is where the candidate ledger starts, so a row can never be
lost between OCR and the ledger (the old ledger started after the row builder
had already discarded rows).

Row segmentation uses multiple anchors, not just dates:

* anchor lines are any line in a date, reference, serial or amount column;
* columns are discovered from content (lines that clearly ARE dates / amounts /
  short references) and then applied by POSITION, so a date that OCR garbled
  (`86+0￥-25`) still anchors its row because it sits in the date column;
* handwriting tilts rows, so each column gets a measured vertical offset
  relative to the densest anchor column before rows are clustered;
* two anchors from the same column never share a row;
* lines that fit no row are kept as their own candidates and classified with a
  reason (header, balance, total, continuation, annotation).

Unreadable fields become "-" on the candidate; the candidate itself survives.
Evidence from every OCR result (all engines, all variants) is attached per field.
"""

import re
import statistics
from dataclasses import asdict, dataclass, field
from decimal import Decimal

from app.accounting.money import parse_amount
from app.documents.ocr import DocumentRepresentation, OcrPage
from app.documents.ocr_contract import OcrLineEvidence, OcrResult
from app.intelligence.parsing import read_handwritten_amount, read_handwritten_date, split_merged_amounts

# Candidate classifications
TRANSACTION = "TRANSACTION"
INFLOW = "INFLOW"
HEADER = "HEADER"
BALANCE = "BALANCE"
TOTAL = "TOTAL"
CONTINUATION = "CONTINUATION"
ANNOTATION = "ANNOTATION"
DOCUMENT_TEXT = "DOCUMENT_TEXT"

# Terminal ledger states
ACCEPTED = "ACCEPTED"
NEEDS_REVIEW = "NEEDS_REVIEW"
REJECTED_WITH_REASON = "REJECTED_WITH_REASON"
NON_TRANSACTION = "NON_TRANSACTION"
UNRESOLVED = "UNRESOLVED"
TERMINAL_STATES = (ACCEPTED, NEEDS_REVIEW, REJECTED_WITH_REASON, NON_TRANSACTION, UNRESOLVED)

# Field roles
REF = "REF"
SERIAL = "SERIAL"
DATE = "DATE"
AMOUNT = "AMOUNT"            # the primary (payment/transaction) amount column
RECEIPT_AMOUNT = "RECEIPT_AMOUNT"
OTHER_AMOUNT = "OTHER_AMOUNT"
PARTICULARS = "PARTICULARS"
CATEGORY = "CATEGORY"
ROW_TEXT = "ROW_TEXT"        # a line spanning several columns (e.g. PaddleOCR row lines)

LETTER_DIGITS = {"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "i": "1", "B": "8", "S": "5", "s": "5",
                 "Z": "2", "z": "2", "g": "9", "q": "9", "y": "4", "t": "7", "T": "7", "A": "4", "b": "6", "G": "6"}

_STRICT_DATE = re.compile(
    r"^\s*\d{1,2}\s*[-/.]\s*(?:\d{1,2}|[A-Za-z]{3,9})\s*[-/.]\s*\d{2,4}\s*$|^\s*\d{4}-\d{2}-\d{2}\s*$"
)
_AMOUNT_LIKE = re.compile(r"^[₹Rs.:\s=]*\(?\d[\d,.]*\)?\s*(?:/-|/=|1-|-|/|\||Cr|Dr)?\s*$", re.IGNORECASE)
HEADER_WORDS = ("particulars", "voucher", "date", "amount", "debit", "credit", "balance", "description",
                "qty", "rate", "s.no", "sr. no", "sr no", "cheque", "narration", "expenditure", "withdrawal",
                "deposit", "member name", "unit", "nos.")
BALANCE_WORDS = ("balance", "carry forward", "carried forward", "brought forward", "b/f", "c/f", "b/d", "c/d",
                 "opening", "closing", "carryforward")
TOTAL_WORDS = ("total", "grand total", "sub total", "subtotal", "net amount", "rupees in words")
INFLOW_WORDS = ("cash with", "withdrawn", "withdrawal", "cash received", "received from", "cash drawn")


@dataclass
class FieldEvidence:
    role: str
    engine: str
    variant: str
    text: str
    confidence: float
    line_id: str
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data


@dataclass
class SourceCandidate:
    candidate_id: str
    page: int
    bbox: tuple[int, int, int, int]
    classification: str
    reason: str
    fields: dict[str, list[FieldEvidence]] = field(default_factory=dict)
    line_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None
    # Filled in by fusion / ledger finalisation.
    status: str | None = None
    status_reason: str = ""
    row_id: str | None = None

    def primary(self, role: str) -> FieldEvidence | None:
        items = self.fields.get(role) or []
        return items[0] if items else None

    def texts(self, role: str) -> list[str]:
        return [f.text for f in self.fields.get(role) or []]

    def all_text(self) -> str:
        return " ".join(f.text for items in self.fields.values() for f in items)

    def has_role(self, role: str) -> bool:
        return bool(self.fields.get(role))

    def add(self, ev: FieldEvidence):
        bucket = self.fields.setdefault(ev.role, [])
        if all(existing.line_id != ev.line_id for existing in bucket):
            bucket.append(ev)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "page": self.page,
            "bbox": list(self.bbox),
            "classification": self.classification,
            "reason": self.reason,
            "parent_id": self.parent_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "row_id": self.row_id,
            "line_ids": list(self.line_ids),
            "fields": {role: [f.to_dict() for f in items] for role, items in self.fields.items()},
        }


@dataclass
class PageLayout:
    page: int
    tabular: bool
    pitch: float
    date_col: tuple[float, float] | None
    ref_col: tuple[float, float] | None
    serial_col: tuple[float, float] | None
    amount_cols: list[tuple[float, float]]
    payment_col_index: int | None
    receipt_col_index: int | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# content kinds
# ---------------------------------------------------------------------------

def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _is_strict_date(text: str) -> bool:
    return bool(_STRICT_DATE.match(text)) and read_handwritten_date(text).status == "FOUND"


def _is_amount_like(text: str) -> bool:
    t = text.strip()
    return bool(_AMOUNT_LIKE.match(t)) and len(_digits(t)) >= 2


def _is_short_ref(text: str) -> bool:
    t = text.strip()
    d = _digits(t)
    return 1 <= len(d) <= 4 and len(t) <= 5 and sum(ch.isalpha() for ch in t) <= 1


def _is_merged_totals(text: str) -> bool:
    """One OCR line holding two grouped totals, e.g. '98,6231,08,300'."""
    return "," in text and len(split_merged_amounts(text)) >= 2


def _has_any(text: str, words, fuzzy: float = 0.0) -> bool:
    """Keyword test; with `fuzzy`, single-word keywords also match OCR-misspelt
    tokens of similar length (handwriting: 'Belence' ~ 'balance')."""
    low = text.lower()
    if any(w in low for w in words):
        return True
    if not fuzzy:
        return False
    from difflib import SequenceMatcher
    tokens = re.findall(r"[a-z]{5,}", low)
    for w in words:
        if " " in w or len(w) < 5:
            continue
        for t in tokens:
            if abs(len(t) - len(w)) <= 3 and SequenceMatcher(None, t, w).ratio() >= fuzzy:
                return True
    return False


def _header_hits(text: str) -> int:
    from difflib import SequenceMatcher
    low = text.lower()
    tokens = re.findall(r"[a-z.]{3,}", low)
    hits = 0
    for w in HEADER_WORDS:
        if w in low or (len(w) >= 6 and any(SequenceMatcher(None, t, w).ratio() >= 0.72 for t in tokens)):
            hits += 1
    return hits


def _cluster_1d(values: list[float], gap: float) -> list[list[float]]:
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] > gap:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    return clusters


def _span(lines) -> tuple[float, float]:
    xs0 = sorted(l.x0 for l in lines)
    xs1 = sorted(l.x1 for l in lines)
    lo = xs0[len(xs0) // 10]
    hi = xs1[min(len(xs1) - 1, (len(xs1) * 9) // 10)]
    return float(lo), float(hi)


def _center(line) -> float:
    return (line.x0 + line.x1) / 2


def _in_col(line, col: tuple[float, float] | None, slack: float = 0.0) -> bool:
    if not col:
        return False
    c = _center(line)
    return col[0] - slack <= c <= col[1] + slack


# ---------------------------------------------------------------------------
# layout discovery
# ---------------------------------------------------------------------------

def discover_layout(page: OcrPage, lines) -> PageLayout:
    width = max(1, page.width)
    notes: list[str] = []
    heights = [l.height for l in lines] or [20]
    med_h = statistics.median(heights)

    date_lines = [l for l in lines if _is_strict_date(l.text)]
    date_col = None
    if len(date_lines) >= 3:
        # Dates cluster in one column; take the densest x cluster.
        clusters = _cluster_1d([_center(l) for l in date_lines], width * 0.05)
        best = max(clusters, key=len)
        members = [l for l in date_lines if min(best) - 1 <= _center(l) <= max(best) + 1]
        if len(members) >= 3:
            lo, hi = _span(members)
            date_col = (lo, hi)
            notes.append(f"date column x={lo:.0f}-{hi:.0f} from {len(members)} readable dates")

    # Amount columns: right-hand clusters of amount-like values.
    amount_lines = [l for l in lines if _is_amount_like(l.text) and _center(l) > width * 0.35
                    and not (date_col and _in_col(l, date_col)) and not _is_merged_totals(l.text)]
    amount_cols: list[tuple[float, float]] = []
    col_sizes: list[int] = []
    for cluster in _cluster_1d([_center(l) for l in amount_lines], width * 0.035):
        if len(cluster) < 2:
            continue
        members = [l for l in amount_lines if min(cluster) - 1 <= _center(l) <= max(cluster) + 1]
        lo, hi = _span(members)
        amount_cols.append((lo, hi))
        col_sizes.append(len(members))
    payment_idx = receipt_idx = None
    if amount_cols:
        payment_idx = max(range(len(amount_cols)), key=lambda i: col_sizes[i])
        right = [i for i in range(len(amount_cols)) if i > payment_idx]
        # A register's cash-received column: a sparser column right of payments.
        if right and len(amount_cols) <= 3:
            receipt_idx = right[-1]
        notes.append(f"{len(amount_cols)} amount column(s); primary at index {payment_idx}")

    ref_col = serial_col = None
    if date_col:
        left_short = [l for l in lines if _is_short_ref(l.text) and l.x1 <= date_col[0] + (date_col[1] - date_col[0]) * 0.3
                      and _center(l) < date_col[0]]
        # Serial and voucher columns sit side by side and their centres blur
        # together; their LEFT edges separate cleanly, so cluster on x0.
        clusters = _cluster_1d([float(l.x0) for l in left_short], width * 0.018)
        cols = []
        for cluster in clusters:
            members = [l for l in left_short if min(cluster) - 1 <= l.x0 <= max(cluster) + 1]
            if len(members) < 3:
                continue
            three = sum(1 for m in members if len(_digits(m.text)) >= 3) / len(members)
            xs0 = sorted(m.x0 for m in members)
            xs1 = sorted(m.x1 for m in members)
            span = (float(xs0[0]), float(xs1[len(xs1) // 2]))
            cols.append((statistics.median(_center(m) for m in members), span, three, len(members)))
        cols.sort(key=lambda c: c[0])
        if cols:
            # The reference/voucher column is the one whose values are mostly >= 3 digits.
            ref_candidates = [c for c in cols if c[2] >= 0.5]
            if ref_candidates:
                ref = max(ref_candidates, key=lambda c: c[3])
                ref_col = ref[1]
                serials = [c for c in cols if c is not ref and c[0] < ref[0]]
                if serials:
                    serial_col = max(serials, key=lambda c: c[3])[1]
            else:
                serial_col = max(cols, key=lambda c: c[3])[1]
        if ref_col:
            notes.append(f"reference column x={ref_col[0]:.0f}-{ref_col[1]:.0f}")
        if serial_col:
            notes.append(f"serial column x={serial_col[0]:.0f}-{serial_col[1]:.0f}")

    return PageLayout(
        page=page.page_number, tabular=False, pitch=med_h * 1.2,
        date_col=date_col, ref_col=ref_col, serial_col=serial_col,
        amount_cols=amount_cols, payment_col_index=payment_idx, receipt_col_index=receipt_idx, notes=notes,
    )


def _anchor_key(line, role: str, layout: PageLayout) -> str:
    """Role plus column index, so ten amount columns are ten distinct anchors."""
    if role in (AMOUNT, RECEIPT_AMOUNT, OTHER_AMOUNT) and layout.amount_cols:
        idx = min(range(len(layout.amount_cols)),
                  key=lambda i: abs(_center(line) - sum(layout.amount_cols[i]) / 2))
        return f"{role}#{idx}"
    return role


def _role_of(line, layout: PageLayout, width: int) -> str | None:
    """Anchor role by position (column), independent of whether OCR read it cleanly."""
    spans_many = (line.x1 - line.x0) > width * 0.45
    if spans_many:
        return ROW_TEXT
    if layout.date_col and _in_col(line, layout.date_col, 4):
        return DATE
    if layout.ref_col and _in_col(line, layout.ref_col, 4):
        return REF
    if layout.serial_col and _in_col(line, layout.serial_col, 4):
        return SERIAL
    # Nearest amount column whose (padded) span contains the line.
    best_idx, best_dist = None, None
    for idx, col in enumerate(layout.amount_cols):
        pad = (col[1] - col[0]) * 0.25 + 4
        if _in_col(line, col, pad) and len(_digits(line.text)) >= 1:
            dist = abs(_center(line) - (col[0] + col[1]) / 2)
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = idx, dist
    if best_idx is None:
        return None
    if best_idx == layout.payment_col_index:
        return AMOUNT
    if best_idx == layout.receipt_col_index:
        return RECEIPT_AMOUNT
    return OTHER_AMOUNT


def _text_role(line, layout: PageLayout) -> str:
    left_edge = min(c[0] for c in (layout.serial_col, layout.ref_col, layout.date_col) if c) if (
        layout.serial_col or layout.ref_col or layout.date_col) else None
    if left_edge is not None and line.x1 <= left_edge + 8:
        return CATEGORY
    return PARTICULARS


# ---------------------------------------------------------------------------
# band construction
# ---------------------------------------------------------------------------

@dataclass
class _Band:
    y: float
    members: list = field(default_factory=list)       # (role, line)
    roles: set = field(default_factory=set)


def _pitch(anchor_by_role: dict[str, list], med_h: float) -> float:
    diffs = []
    for lines in anchor_by_role.values():
        ys = sorted(l.y_center for l in lines)
        diffs += [b - a for a, b in zip(ys, ys[1:]) if b - a > med_h * 0.5]
    return statistics.median(diffs) if diffs else med_h * 1.2


def build_page_candidates(page: OcrPage, id_prefix: str | None = None) -> tuple[list[SourceCandidate], PageLayout]:
    width = max(1, page.width)
    primary = list(page.lines)
    prefix = id_prefix or f"c{page.page_number}"
    if not primary:
        return [], PageLayout(page.page_number, False, 0.0, None, None, None, [], None, None, ["no text on page"])

    layout = discover_layout(page, primary)
    med_h = statistics.median(l.height for l in primary)

    anchor_roles = (DATE, REF, SERIAL, AMOUNT, RECEIPT_AMOUNT, OTHER_AMOUNT)
    roles = {id(l): _role_of(l, layout, width) for l in primary}
    anchors: dict[str, list] = {}
    keys: dict[int, str] = {}
    for l in primary:
        role = roles[id(l)]
        if role in anchor_roles:
            key = _anchor_key(l, role, layout)
            keys[id(l)] = key
            anchors.setdefault(key, []).append(l)

    pitch = _pitch(anchors, med_h) if anchors else med_h * 1.2
    layout.pitch = round(pitch, 1)

    # Per-column vertical offset relative to the densest anchor column (tilt).
    offsets: dict[str, float] = {}
    if anchors:
        base_role = max(anchors, key=lambda r: len(anchors[r]))
        base_ys = sorted(l.y_center for l in anchors[base_role])
        offsets[base_role] = 0.0
        for role, lines in anchors.items():
            if role == base_role:
                continue
            deltas = []
            for l in lines:
                nearest = min(base_ys, key=lambda y: abs(y - l.y_center))
                if abs(l.y_center - nearest) < pitch * 0.5:
                    deltas.append(l.y_center - nearest)
            offsets[role] = statistics.median(deltas) if deltas else 0.0

    # Cluster anchors into bands; one anchor per column per band.
    bands: list[_Band] = []
    ordered = sorted(((l.y_center - offsets.get(keys[id(l)], 0.0), keys[id(l)], l)
                      for role_lines in anchors.values() for l in role_lines), key=lambda t: t[0])
    for y, key, line in ordered:
        role = key
        best = None
        for band in bands[-4:]:
            if abs(band.y - y) <= pitch * 0.45 and role not in band.roles:
                if best is None or abs(band.y - y) < abs(best.y - y):
                    best = band
        if best is None:
            best = _Band(y=y)
            bands.append(best)
        best.members.append((roles[id(line)], line))
        best.roles.add(key)
        ys = [ln.y_center - offsets.get(keys.get(id(ln), r), 0.0) for r, ln in best.members]
        best.y = sum(ys) / len(ys)
    bands.sort(key=lambda b: b.y)

    # Attach every non-anchor line of the primary result to a band, by its
    # own measured offset per horizontal zone; anything that fits nowhere is
    # kept as a stand-alone candidate.
    leftovers = []
    others = [l for l in primary if roles[id(l)] not in anchor_roles]
    zone = lambda l: int(_center(l) // (width / 6))
    zone_offsets: dict[int, float] = {}
    if bands:
        for z in {zone(l) for l in others}:
            deltas = []
            for l in others:
                if zone(l) != z:
                    continue
                nearest = min(bands, key=lambda b: abs(b.y - l.y_center))
                if abs(l.y_center - nearest.y) < pitch * 0.5:
                    deltas.append(l.y_center - nearest.y)
            zone_offsets[z] = statistics.median(deltas) if len(deltas) >= 3 else 0.0
    for l in others:
        role = roles[id(l)] or _text_role(l, layout)
        if not bands:
            leftovers.append((role, l))
            continue
        y = l.y_center - zone_offsets.get(zone(l), 0.0)
        nearest = min(bands, key=lambda b: abs(b.y - y))
        if abs(nearest.y - y) <= pitch * 0.55:
            nearest.members.append((role, l))
        else:
            leftovers.append((role, l))

    # Leftover lines grouped into their own visual rows.
    from app.documents.ocr import group_rows
    leftover_groups = group_rows([l for _, l in leftovers]) if leftovers else []
    role_by_line = {id(l): r for r, l in leftovers}
    for group in leftover_groups:
        ys = [l.y_center for l in group]
        band = _Band(y=sum(ys) / len(ys))
        for l in group:
            band.members.append((role_by_line[id(l)], l))
        bands.append(band)
    bands.sort(key=lambda b: b.y)

    tx_like = sum(1 for b in bands if len({r for r, _ in b.members} & {DATE, REF, AMOUNT, OTHER_AMOUNT}) >= 2
                  or sum(1 for r, _ in b.members if r in (AMOUNT, OTHER_AMOUNT)) >= 3)
    layout.tabular = tx_like >= 3
    if layout.tabular:
        layout.notes.append(f"tabular layout: {tx_like} rows carry two or more of date/reference/amount")

    candidates: list[SourceCandidate] = []
    for index, band in enumerate(bands, start=1):
        lines = [l for _, l in band.members]
        x0 = min(l.bbox[0] for l in lines); y0 = min(l.bbox[1] for l in lines)
        x1 = max(l.bbox[2] for l in lines); y1 = max(l.bbox[3] for l in lines)
        cand = SourceCandidate(
            candidate_id=f"{prefix}_{index:03d}", page=page.page_number, bbox=(x0, y0, x1, y1),
            classification="", reason="", line_ids=[l.line_id for l in lines],
        )
        for role, l in sorted(band.members, key=lambda m: m[1].x0):
            cand.add(FieldEvidence(role, l.engine, l.variant, l.text, l.confidence, l.line_id, tuple(l.bbox)))
        candidates.append(cand)

    _attach_other_results(page, candidates, layout, offsets, pitch)
    _classify(candidates, layout, page)
    return candidates, layout


def _attach_other_results(page: OcrPage, candidates: list[SourceCandidate], layout: PageLayout,
                          offsets: dict[str, float], pitch: float):
    """Attach lines from every non-primary OCR result as extra field evidence."""
    primary_ids = {lid for c in candidates for lid in c.line_ids}
    width = max(1, page.width)
    for result in page.evidence:
        for ev in result.lines:
            if ev.line_id in primary_ids:
                continue
            role = _role_of(ev, layout, width) or _text_role(ev, layout)
            y = ev.y_center - offsets.get(_anchor_key(ev, role, layout), 0.0)
            best, best_overlap = None, 0.0
            for cand in candidates:
                top = max(cand.bbox[1], ev.bbox[1]); bottom = min(cand.bbox[3], ev.bbox[3])
                overlap = max(0, bottom - top) / max(1, min(ev.height, cand.bbox[3] - cand.bbox[1]))
                centre_y = (cand.bbox[1] + cand.bbox[3]) / 2
                if overlap > best_overlap or (overlap == best_overlap and best is not None and
                                              abs(centre_y - y) < abs((best.bbox[1] + best.bbox[3]) / 2 - y)):
                    best, best_overlap = cand, overlap
            if best is not None and best_overlap >= 0.5:
                best.add(FieldEvidence(role, ev.engine, ev.variant, ev.text, ev.confidence, ev.line_id, tuple(ev.bbox)))


def _readable_amount(cand: SourceCandidate) -> bool:
    return bool(amount_readings(cand, roles=(AMOUNT, RECEIPT_AMOUNT)))


def anchors_readable(cand: SourceCandidate) -> bool:
    """A readable reference (>= 2 digits) or a parseable date."""
    return any(len(d) >= 2 for d, _ in ref_readings(cand)) or bool(date_readings(cand))


def _classify(candidates: list[SourceCandidate], layout: PageLayout, page: OcrPage):
    tx_indices = []
    # Column headers sit above the first row that has a readable reference or date.
    first_data_y = min((c.bbox[1] for c in candidates if anchors_readable(c)), default=None)
    for idx, cand in enumerate(candidates):
        text = cand.all_text()
        anchors = {r for r in (DATE, REF, AMOUNT, RECEIPT_AMOUNT, OTHER_AMOUNT) if cand.has_role(r)}
        header_hits = _header_hits(text)
        merged_totals = any(_is_merged_totals(t) for t in
                            cand.texts(AMOUNT) + cand.texts(RECEIPT_AMOUNT) + cand.texts(OTHER_AMOUNT))
        above_data = first_data_y is not None and cand.bbox[3] <= first_data_y + layout.pitch * 0.5

        if _has_any(text, BALANCE_WORDS, fuzzy=0.72) and not cand.has_role(REF) and header_hits < 2:
            cand.classification, cand.reason = BALANCE, "balance / carry-forward line"
        elif (header_hits >= 2 or (header_hits >= 1 and above_data)) and not anchors_readable(cand):
            cand.classification, cand.reason = HEADER, f"column header text ({header_hits} header word(s))"
        elif _has_any(text, TOTAL_WORDS) or merged_totals:
            cand.classification, cand.reason = TOTAL, ("written totals" if merged_totals else "total line")
        elif layout.tabular and cand.has_role(RECEIPT_AMOUNT) and not cand.has_role(AMOUNT):
            cand.classification, cand.reason = INFLOW, "amount only in the cash-received column"
        elif layout.tabular and _has_any(text, INFLOW_WORDS) and not cand.has_role(REF):
            cand.classification, cand.reason = INFLOW, "narration describes cash received/withdrawn"
        elif anchors:
            cand.classification = TRANSACTION
            cand.reason = "row carries " + ", ".join(sorted(a.lower() for a in anchors))
            tx_indices.append(idx)
        else:
            cand.classification, cand.reason = ANNOTATION, "text with no date, reference or amount"

    # Everything above the table's column header is document metadata
    # (customer id, account number, period), not a transaction row.
    headers = [c for c in candidates if c.classification == HEADER]
    if headers and first_data_y is not None:
        table_top = max((h.bbox[1] for h in headers if h.bbox[1] < first_data_y), default=None)
        if table_top is not None:
            for cand in candidates:
                if cand.bbox[3] <= table_top and cand.classification in (TRANSACTION, ANNOTATION):
                    cand.classification = DOCUMENT_TEXT
                    cand.reason = "document header text above the table's column headings"
                    if cand in [candidates[i] for i in tx_indices]:
                        tx_indices.remove(candidates.index(cand))

    # Text-only rows directly under a transaction are wrapped narration.
    for idx, cand in enumerate(candidates):
        if cand.classification != ANNOTATION or idx == 0:
            continue
        prev = candidates[idx - 1]
        gap = cand.bbox[1] - prev.bbox[3]
        if prev.classification in (TRANSACTION, INFLOW, CONTINUATION) and gap <= max(layout.pitch, 1) * 0.9:
            parent = prev if prev.classification != CONTINUATION else next(
                (c for c in candidates if c.candidate_id == prev.parent_id), prev)
            cand.classification = CONTINUATION
            cand.parent_id = parent.candidate_id
            cand.reason = f"wrapped narration of {parent.candidate_id}"

    # A total written below the last transaction with only amounts is a total.
    if tx_indices:
        last = candidates[tx_indices[-1]]
        for cand in candidates:
            if cand.classification == TRANSACTION and cand.bbox[1] > last.bbox[3] and not (
                    cand.has_role(REF) or cand.has_role(DATE)):
                cand.classification, cand.reason = TOTAL, "amount-only line below the last transaction"


def build_candidates(rep: DocumentRepresentation) -> tuple[list[SourceCandidate], list[PageLayout]]:
    candidates: list[SourceCandidate] = []
    layouts: list[PageLayout] = []
    for page in rep.pages:
        page_cands, layout = build_page_candidates(page)
        candidates += page_cands
        layouts.append(layout)
    return candidates, layouts


# ---------------------------------------------------------------------------
# field readings from OCR evidence (used for fusion and for local extraction)
# ---------------------------------------------------------------------------

def _letters_to_digits(text: str) -> str:
    return "".join(LETTER_DIGITS.get(ch, ch) for ch in text)


def ref_readings(cand: SourceCandidate) -> list[tuple[str, FieldEvidence]]:
    out = []
    for ev in cand.fields.get(REF) or []:
        digits = _digits(_letters_to_digits(ev.text))
        if digits:
            out.append((digits, ev))
    return out


def date_readings(cand: SourceCandidate) -> list[tuple[object, FieldEvidence]]:
    out = []
    for ev in cand.fields.get(DATE) or []:
        reading = read_handwritten_date(ev.text)
        for value in (reading.candidates if reading.status != "MISSING" else []):
            out.append((value, ev))
    return out


def amount_readings(cand: SourceCandidate, roles=(AMOUNT,)) -> list[tuple[Decimal, FieldEvidence]]:
    """Every value an amount line supports (the '/-' suffix artefact yields two)."""
    out = []
    for role in roles:
        for ev in cand.fields.get(role) or []:
            if ev.engine == "pdf_text":
                reading = parse_amount(ev.text)
                values = [reading.value] if reading.found else []
            else:
                reading = read_handwritten_amount(ev.text)
                values = list(reading.candidates or ([] if reading.value is None else [reading.value]))
            for v in values:
                if v is not None:
                    out.append((Decimal(v), ev))
    # Row-spanning lines (PaddleOCR reads a handwritten row as one line): the
    # trailing token, when the line reaches the amount column, is amount evidence.
    for ev in cand.fields.get(ROW_TEXT) or []:
        tokens = ev.text.split()
        if tokens and AMOUNT in roles:
            reading = read_handwritten_amount(tokens[-1])
            for v in reading.candidates or []:
                if v is not None:
                    out.append((Decimal(v), FieldEvidence(AMOUNT, ev.engine, ev.variant, tokens[-1],
                                                          ev.confidence, ev.line_id, ev.bbox)))
    return out


def band_digit_text(cand: SourceCandidate) -> str:
    return _digits(_letters_to_digits(cand.all_text()))
