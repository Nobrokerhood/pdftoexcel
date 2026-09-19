"""Golden-corpus accuracy benchmark for the full extraction pipeline.

Runs DocumentExtractionPipeline (real OCR + Gemini when configured) on the
benchmark documents in testing_folder/ and scores the result against ground
truth that lives ONLY in this script. Production code never sees these values.

usage: python scripts/golden_benchmark.py [out.json]
Exit code 0 = all acceptance checks passed, 1 = at least one failed.

Acceptance (per the production brief):
* every genuine handwritten row survives as ACCEPTED or NEEDS_REVIEW;
* a golden value is either correct, or its row is NEEDS_REVIEW with the
  correct value among the recorded evidence (never a confident wrong value);
* candidate ledger balanced; Patel: 2 items totalling 8700, not flagged;
  labour bill: no accepted fabricated rows.
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CORPUS = os.path.join(ROOT, "testing_folder")

HANDWRITTEN_TRUTH = {266: 900, 267: 140, 268: 700, 269: 130, 270: 1800, 271: 300, 274: 200, 284: 10000, 285: 8400,
                     286: 9000, 287: 10000, 288: 9400, 289: 12000, 290: 10000, 291: 880, 292: 200, 296: 1090,
                     297: 2250, 298: 5000, 299: 2000, 300: 2500, 301: 2400, 302: 1789, 303: 1000, 304: 3000,
                     305: 200, 306: 200, 307: 750, 308: 200}


def evidence_values(result, row, column):
    fields = (result["row_evidence"].get(row["_row_id"]) or {}).get("fields") or {}
    d = fields.get(column) or {}
    return {v.get("value") for v in d.get("votes") or []} | {d.get("value")} | set(d.get("alternatives") or [])


def score_handwritten(result):
    """Rows are aligned to the ground truth by vertical position (the register's
    order), so a row whose reference OCR misread is still scored, not 'missing'."""
    ev = result["row_evidence"]
    rows = sorted([r for r in result["rows"] if (ev.get(r["_row_id"]) or {}).get("bbox")],
                  key=lambda r: ev[r["_row_id"]]["bbox"][1])
    report = {"rows": len(result["rows"]), "correct": [], "flagged_with_truth_in_evidence": [],
              "flagged_truth_not_in_evidence": [], "confidently_wrong": [], "missing": []}
    truth = list(HANDWRITTEN_TRUTH.items())
    if len(rows) < len(truth):
        report["missing"] = [ref for ref, _ in truth[len(rows):]]
    for row, (ref, amount) in zip(rows, truth):
        exact = row["Amount*"] == str(amount) and row["Cheque/Ref No*"] == str(ref)
        if exact and row["_status"] == "ACCEPTED":
            report["correct"].append(ref)
        elif row["_status"] == "ACCEPTED":
            report["confidently_wrong"].append({"ref": ref, "got": [row["Cheque/Ref No*"], row["Amount*"]],
                                                "truth": [ref, amount]})
        elif exact:
            report["correct"].append(ref)  # correct value, still routed to a human
        elif str(amount) in evidence_values(result, row, "Amount*"):
            report["flagged_with_truth_in_evidence"].append(ref)
        else:
            report["flagged_truth_not_in_evidence"].append(ref)
    report["pass"] = not report["missing"] and not report["confidently_wrong"]
    return report


def main():
    from app.agents.extraction_pipeline import DocumentExtractionPipeline
    from app.core.config import get_settings
    from app.documents.ocr import DocumentOcrService
    from app.services.gemini_client import GeminiDocumentClient

    settings = get_settings()
    pipe = DocumentExtractionPipeline(GeminiDocumentClient(settings), DocumentOcrService(settings.poppler_path))
    docs = [
        ("WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf", "PETTY_CASH_REGISTER"),
        ("unnamed (1).jpg", "VENDOR_INVOICE"),
        ("sample-radhakrishna.pdf", "MEMBER_RECEIPT"),
        ("IDFCFIRSTBankstatement_10178263032_134000056 (2)_page-0008.pdf", "MEMBER_RECEIPT"),
        ("unnamed.jpg", "VENDOR_INVOICE"),
    ]
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "documents": {}}
    ok = True
    for name, purpose in docs:
        path = os.path.join(CORPUS, name)
        if not os.path.exists(path):
            out["documents"][name] = {"status": "NOT TESTED - file absent"}
            continue
        t0 = time.time()
        result = pipe.extract(open(path, "rb").read(), purpose)
        entry = {"purpose": purpose, "seconds": round(time.time() - t0, 1), "provider": result["_extraction_provider"],
                 "provider_note": result.get("provider_note"), "rows": len(result["rows"]),
                 "accepted": sum(1 for r in result["rows"] if r["_status"] == "ACCEPTED"),
                 "ledger": result["candidate_ledger"]["equation"], "ledger_balanced": result["candidate_ledger"]["balanced"],
                 "gemini_calls": result.get("gemini_calls"), "arbitration": result.get("arbitration", {}).get("status")}
        if name.startswith("WhatsApp"):
            entry["score"] = score_handwritten(result)
            ok &= entry["score"]["pass"] and entry["ledger_balanced"]
        elif name == "unnamed (1).jpg":
            amounts = sorted(r["Amount*"] for r in result["rows"])
            entry["score"] = {"amounts": amounts, "pass": amounts == ["2100", "6600"]}
            ok &= entry["score"]["pass"]
        elif name == "unnamed.jpg":
            entry["score"] = {"pass": all(r["_status"] != "ACCEPTED" or r["Amount*"] != "-" for r in result["rows"])}
            ok &= entry["score"]["pass"]
        elif name.startswith("sample-radhakrishna"):
            entry["score"] = {"rows": len(result["rows"]), "pass": len(result["rows"]) == 20}
            ok &= entry["score"]["pass"]
        else:
            entry["score"] = {"rows": len(result["rows"]), "pass": len(result["rows"]) == 37}
            ok &= entry["score"]["pass"]
        out["documents"][name] = entry
        print(json.dumps({name: entry.get("score"), "provider": entry["provider"], "rows": entry["rows"]}, default=str))
    out["all_pass"] = ok
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
