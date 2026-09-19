# Production Readiness Report: NoBrokerHood Accounting AI

**Final status: NOT_PRODUCTION_READY**

The code now meets the brief's correctness and safety requirements, with evidence below. Production cannot pass the acceptance gate yet, because of four infrastructure and configuration blockers that code cannot fix (Section 20). Each was measured, not assumed.

| | |
|---|---|
| Date | 2026-09-19 |
| Baseline | `bb3575f` + uncommitted work (snapshot object `f270706f`) |
| Delivered commits (branch `production-accuracy-ux-performance`, fast-forwarded to `main`) | `2784932` pipeline rebuild; `55e4c05` merge of `main` (keeps its login page and domain rule); `1fccf46` Sheets/amount-column fixes; `7f62ee6`, `b3b3605`, `fe1621e` memory-safety fixes |
| Deployed | Production `/config/version` reports commit `fe1621e6da55ece0c124489e014a963b9f7c71e6`, branch `main` |
| Frontend | GitHub Pages serves the new UI (polling and review panel present, fake timer absent) |

## 1. Executive summary

The pipeline was rebuilt around source evidence:
- **One OCR contract.** All engines run through a single orchestrator.
- **Evidence kept.** Every engine and variant result is kept on a shared canonical page image.
- **Candidate ledger.** It is built from OCR before any filtering, and every source region ends in exactly one state.
- **Rows.** Row construction uses multiple anchors, and rows carry immutable ids.
- **Model use.** Gemini works under strict per-purpose contracts, and a blind per-cell visual read acts as the arbiter.
- **Verification and repair.** Verification is keyed by row id and never defaults to PASSED. Repair works field by field.
- **Accounting and export.**
  - Amount and date parsing is strict.
  - Reconciliation keeps source-written and calculated values apart.
  - The XLSX is exactly 12 columns and formula-safe.
  - No placeholder rows are fabricated.
- **Human review.** The review workflow is human-resolvable, and approval is gated on resolved items.
- **Operations.**
  - Jobs run in the background with a real stage trace.
  - Job state persistence is never truncated.
  - Sheets usage respects quotas.
- **Security.** The security holes are closed.

**Test suite:** 323 passed, 0 failed. Inside the production Docker image, 120 passed, including the golden corpus with real OCR on Linux.

The real local end-to-end run (real Drive, Sheets and OCR) completed for all 5 golden documents, including human approval. The output workbook landed in Drive Output and the source moved to Completed.

**What still blocks production:**
- The Render instance has 512 MB of memory. OCR cannot run in it.
- The Gemini project has hit its spend cap, both locally and in production.
- Production runs with `ENVIRONMENT` unset.
- The production `SESSION_SECRET` is the same value as in the developer `.env`.

## 2. Architecture before

See `docs/PRODUCTION_FORENSIC_AUDIT.md`: 60+ findings, 30 of them BLOCKER. In brief:
- RapidOCR was hardcoded in three places, and PaddleOCR never ran.
- Alternative OCR output was discarded, and digital-PDF geometry was fabricated.
- Rows were anchored only on dates, so un-anchored rows were silently dropped. The ledger started only after that loss.
- PETTY_CASH_REGISTER was sent the vendor schema.
- Verification defaulted to PASSED, and repair rewrote the whole record.
- NEEDS_REVIEW could never be approved.
- The Excel writer turned `Rs.900` into 0.9.
- The UI used a fake progress timer, and processing ran synchronously inside the upload request.
- Uploads accepted a `SYNTHETIC` bypass, and the legacy Gemini endpoints had no authentication.

## 3. Architecture after

```
upload -> signature/MIME/extension/pixel/page validation -> Drive Incoming -> background job
  -> canonical page images (one per page; OCR, Gemini and crops share its pixels)
  -> OcrOrchestrator: RapidOCR (+ PaddleOCR + variant on difficult pages, memory permitting);
     every result retained; field-aware scoring picks the primary line set
  -> source candidates (multi-anchor rows) = start of the candidate ledger
  -> Gemini purpose contract (strict normalisation; violations -> OCR-only rows + review)
  -> fusion by independent evidence groups -> blind per-cell arbitration
  -> row-id verification -> bounded field-level repair -> mapping (non-blocking)
  -> deterministic validation + reconciliation -> review items -> human review
  -> approval gate (no open blocking items, ledger balanced, valid mandatory fields)
  -> 12-column XLSX + evidence/reconciliation/ledger/audit sheets -> Drive Output
  -> source moved to Completed
```

## 4–7. Blockers found → root cause → fix → proving tests

| Blocker | Root cause | Fix | Proving test(s) |
|---|---|---|---|
| Handwritten rows lost (270, 299, 300, 304, 306, 308; refs of 284/287/296) | Rows anchored only on dates; leftovers discarded | `source_candidates.py`: rows anchored on any of date/ref/serial/amount, placed by column position, with per-column tilt offsets | `test_golden_corpus::test_every_genuine_handwritten_row_exists_as_a_source_candidate` (real OCR, 29/29) |
| Ledger blind to loss | Counted rows after the builder had already dropped some | Ledger over OCR candidates, one terminal state each | `test_degraded_mode_exports_all_29_rows_with_a_balanced_ledger` |
| Petty cash got the vendor schema; list and flat-dict replies mishandled | Shared prompt; no shape validation | `gemini_contract.py`: per-purpose prompts and models; strict `normalize_response` | `test_money_dates_contract` (12 contract cases) |
| Engines hardcoded / PaddleOCR unused / evidence discarded | No contract, no orchestrator | `ocr_contract.py`, `ocr_engines.py`, `ocr_orchestrator.py` | `test_no_module_outside_the_orchestration_layer_instantiates_an_ocr_engine`; `test_ocr_evidence_retains_every_engine_and_variant` |
| Coordinate mismatch (150 vs 180 DPI) | Separate renders for OCR and Gemini | One canonical image per page | `test_ocr_evidence_retains_every_engine_and_variant` (sizes asserted equal) |
| Fabricated digital-PDF boxes | Synthetic `(0, i*25, 800, …)` | pdfplumber word segments, geometry REAL | `test_requirement_10_digital_pdf_bypass_in_real_workflow` |
| Gemini copying OCR hints counted as corroboration (found live: 206, 568, 801) | Page read treated as independent of the OCR text it was shown | Independence groups; blind crop reads | `test_page_read_anchored_to_ocr_is_not_independent` |
| Rate/qty column "verified" as the amount (found by golden benchmark: Patel 330) | Every numeric column voted on the amount | Only the primary amount column votes; others may only support a proposed value | `test_other_numeric_columns_cannot_verify_an_amount` |
| Default PASSED; results matched by position | Adapter defaults; `row_N` name matching | `adapt_verification` keyed by row id; missing status becomes NEEDS_REVIEW | `test_verifier_sends_every_row_and_attributes_only_by_row_id`; `test_missing_overall_status_is_never_passed` |
| Repair rewrote whole record; edits restored by position | Whole-record regeneration | Field-id requests only; edited fields never touched | `test_repair_only_fills_the_requested_field_and_normalizes_it`; `test_grid_save_matches_rows_by_id_not_position` |
| NEEDS_REVIEW never approvable | Approval required Gemini PASSED | Review items plus audited resolutions; gate on unresolved blocking items | `test_genuine_blockers_still_prevent_approval` (confirm → approve) |
| `Rs.900`→0.9, `2500 (105)`→2500105, `1.500,00`→1.5; mixed date types | Regex strip; ad-hoc dates | `money.py`, `dates.py`, strict `nbh_cells` | `test_export_amounts_are_parsed_strictly`; `test_export_refuses_ambiguous_or_missing_amounts`; `test_export_dates_follow_one_policy` |
| Placeholder row fabricated at export | `rows or [data]` fallback | Removed; zero rows block approval | `test_zero_rows_never_produce_a_workbook`; `test_zero_rows_from_gemini_is_never_an_approvable_empty_workbook` |
| Formula injection | Strings written as formulas | Forced string cells; RAW Sheets writes | `test_formula_like_source_text_is_stored_as_text`; `test_row_update_is_a_single_batched_raw_write` |
| Fake progress | 7-second timer | Backend stage trace plus polling | Timer removed (0 occurrences on the deployed page) |
| Synchronous processing; per-process state with 2 workers | In-request pipeline | Bounded background runner; 1 worker | `test_background_runner_returns_before_work_finishes` |
| Job state truncated at 45 k characters → stub | Raw JSON in one cell | zlib+base64, with Drive fallback; never a stub | `test_large_job_state_round_trips_without_truncation` |
| **Sheets read quota exhausted by one job** (found in real E2E: 500s) | Every call re-opened the sheet | Cached handles and rows; queued, coalescing, 429-tolerant writer | `test_every_sheets_call_failing_never_breaks_upload_or_processing`; real E2E 5/5 |
| `SYNTHETIC` bypass; unauthenticated Gemini/KB/audit endpoints | Test hooks and missing auth | Removed; session required | `test_synthetic_signature_bypass_is_gone`; `test_legacy_tools_require_a_session` |
| **Production OOM crash** (found in production) | PaddleOCR / RapidOCR load exceeds the container limit | Memory-aware profile; OCR skipped and uploads refused below the minimum | `test_undersized_instance_refuses_uploads_instead_of_crashing`; production check (Section 19) |
| Undeclared `pypdf`/`pdfplumber`; unpinned dependencies | Requirements drift | Everything pinned to verified versions | Docker build plus in-image tests |

## 8. Golden corpus results

Every run below was degraded mode, because Gemini is spend-capped. Source: `scripts/golden_benchmark.py`.

| Document | Result | Pass? |
|---|---|---|
| Handwritten register (29 truth rows) | 29/29 rows present; **0 confidently wrong**. 11 exactly correct; 14 flagged NEEDS_REVIEW with the true value in their evidence; 4 flagged without it (297, 300, 302, 307, which need the model or arbitration) | Pass (zero loss, zero confident errors) |
| Patel invoice | 2 rows, 6600 + 2100 = 8700, not flagged as incorrect | Pass |
| IDFC statement | 37/37 rows | Pass |
| Labour bill (`unnamed.jpg`) | 0 rows, no fabricated rows | Pass |
| Radhakrishna | 0 exported rows; its 20 member rows are visible UNRESOLVED candidates (no date or ref anchor; needs the model) | **Fail in degraded mode** |

**NOT TESTED: accuracy with Gemini plus blind cell arbitration.** The project's key is spend-capped. Earlier live runs in this session showed the page-level read varies between runs (row-shifted once). Cell arbitration never completed a full live run.

## 9. Candidate reconciliation results (real OCR)

| Document | Ledger |
|---|---|
| Handwritten | 36 = 0 accepted + 29 needs review + 0 rejected + 6 non-transaction + 1 unresolved (balanced) |
| Patel | 10 candidates, balanced |
| IDFC | 58 = 37 + 19 + 2 (balanced) |
| Radhakrishna | 28 (balanced; 20 unresolved) |
| Labour | 15, all non-transaction (balanced) |

## 10. OCR engine comparison (handwritten page, 200 DPI)

| Engine / variant | Lines | Mean confidence | Time | Field-aware score |
|---|---|---|---|---|
| RapidOCR, canonical | 180 | 0.842 | 5.3 s | 209.8 (selected) |
| RapidOCR, CLAHE variant | 180 | 0.835 | 5.3 s | 209.1 |
| PaddleOCR | 82 | 0.645 | 5.8 s | 77.6 (reads whole rows as one line) |

PaddleOCR still adds independent evidence. For example, on row 302 it read `189` where RapidOCR's CLAHE variant read `1787`.

## 11–12. Gemini and verification behaviour

**Gemini:**
- Purpose prompts carry OCR line ids.
- Replies go through strict normalisation. The list, flat, malformed and empty cases are tested.
- Failure injection passes for 429 (spend cap, daily quota, rate limit), 5xx, timeout, malformed JSON and wrong shape. Every case degrades to OCR rows plus NEEDS_REVIEW, and nothing is fabricated.

**Verification:**
- Keyed by row id; unknown ids are ignored.
- Uncovered rows are UNVERIFIED.
- A model can downgrade a row but never upgrade it.

## 13. Accounting reconciliation

Golden test (`test_handwritten_benchmark_full_regression`), with nothing altered in the source:

| Check | Source | Calculated | Difference |
|---|---|---|---|
| Transaction sum vs written expenditure | 98623 | 96429 | reported |
| Closing balance | 10174 | 10177 | **₹3** |
| Opening deficit vs derived adjustment | 1714 | 2194 | **₹480** |

Both discrepancies surface as non-blocking warnings.

## 14. XLSX validation

- **Primary sheet:** exactly 12 NBH columns, in order, for every purpose.
- **Amounts:** numeric only when unambiguous.
- **Dates:** real `DD-MM-YYYY` dates; missing values are `-`.
- **Text cells:** forced to string type (formula-safe).
- **Supporting sheets:** Evidence, Reconciliation, Source Candidates, Review Audit, Summary, and Vendor Detail for vendor invoices.
- **Verified by:** the real E2E workbook, read back from Drive.

## 15. Drive validation (real, local E2E)

- Source uploaded to Incoming.
- On approval, the workbook was written to Output (`NBH_VENDOR_BILL_V1_f20f765b-….xlsx`) and the source moved to Completed. Confirmed independently through the Drive API.
- All 5 jobs restored from Sheets in a fresh process.

## 16. Security validation

- Closed and tested:
  - the `SYNTHETIC` bypass;
  - unauthenticated legacy, Knowledge Bot and audit endpoints;
  - decompression and page bounds;
  - formula injection (XLSX and Sheets);
  - quoted `Content-Disposition` headers;
  - Drive errors no longer carry folder ids or service-account identity;
  - the live capability probe is admin-only;
  - no secrets in the committed diff (scanned).
- **Open:** see Section 20, items 3 and 4.

## 17. Performance (real local E2E)

| Document | Wall time | OCR + extraction | Notes |
|---|---|---|---|
| Handwritten | 44.7 s | 31.8 s | three OCR reads |
| IDFC | 18.4 s | 14.6 s | |
| Patel | 12.8 s | — | |
| Labour | 12.3 s | — | |
| Radhakrishna | 6.2 s | 1.9 s | digital text |

- Upload responses return in 3–7 s (processing continues in the background).
- Peak memory in the production image: 516 MB with RapidOCR loaded, 830 MB with PaddleOCR added, 1872 MB for the full ensemble on the handwritten page, 837 MB for RapidOCR-only at 200 DPI.

## 18. Production deployment version

`fe1621e6da55ece0c124489e014a963b9f7c71e6` (branch `main`, service `pdftoexcel-846x`), as reported by `/config/version`.

## 19. Production E2E results (real, deployed service)

**Passed:**
- Deployed version is verified.
- Health is stable.
- `/auth/me` returns 401 without a token and 200 with a session.
- The legacy tools return 401 without a session.
- The live probe reports Drive READY, Sheets READY, Poppler READY and OpenCV READY.
- The frontend serves the new UI with no fake timer.
- The capability report returns an honest NOT_READY in 3 s. Before the fix it crashed the instance twice (503 → restart, observed).

**Not possible:**
- A document upload returns 503: *"This server instance has 512 MB of memory; document OCR needs at least 1024 MB."* By design, the instance refuses rather than being killed mid-job.
- **Processing, review, approval and Drive-output E2E in production is therefore NOT TESTED.** Blocked by items 1 and 2 below; the same flow passed locally against real Drive and Sheets.

## 20. Remaining limitations and exact blocking evidence

1. **Render instance memory is 512 MB (BLOCKER).**
   - *Evidence:* `/config/runtime` shows `container_limit_mb: 512`. RapidOCR alone peaks at ~516 MB.
   - *Action (billing):* upgrade the instance.
     - At least 1 GB for RapidOCR-only (DEGRADED).
     - At least 2.5 GB, plus `OCR_ENSEMBLE=auto`, for the full ensemble.
2. **The Gemini project is at its monthly spending cap, both locally and in production (BLOCKER).**
   - *Evidence:* production live probe returns `GeminiSpendCapError`.
   - *Action:* raise the cap or configure a working key. Then run `scripts/golden_benchmark.py` and the production E2E.
3. **`ENVIRONMENT` is not `production` on Render (BLOCKER).**
   - *Evidence:* the capability report shows `"environment":"development"`; `/docs` and `/openapi.json` return 200 publicly; the production config guards are not enforced.
   - *Action:* set `ENVIRONMENT=production` in Render.
4. **The production `SESSION_SECRET` equals the developer `.env` value (BLOCKER, security).**
   - *Evidence:* a token signed with the local value was accepted by production `/auth/me`.
   - *Action:* rotate it in Render to a value that exists nowhere else.
   - Also rotate the service-account key and the OAuth client secret that were pasted in chat earlier.
5. **Radhakrishna-style member lists need the model.** Degraded mode keeps their rows as UNRESOLVED; nothing is lost.
6. **Handwriting.** Values like 307 (750 read as 150 by both engines) can only be corrected by a human or by visual arbitration. They are always flagged (CONFLICT or NEEDS_REVIEW), never silently accepted.
7. **Not tested:**
   - live accuracy with Gemini plus arbitration;
   - production document processing;
   - a Render Docker deployment (the service uses Render's native runtime; the Dockerfile was built and tested locally).

### Acceptance gate

| Gate item | Status |
|---|---|
| Audit, zero silent loss, ledger-first, OCR contract, evidence retained, robust rows | Met |
| Purpose schemas, strict normalisation, coordinates, PDF geometry | Met |
| Row-id verification, no default PASSED, field-safe repair, immutable edits | Met |
| Human review, no fake progress, 12 columns, no fabricated rows | Met |
| Amounts, dates, reconciliation, discrepancies preserved | Met |
| Drive E2E, Sheets resilience, Gemini and OCR failure modes, security fixes | Met (locally) |
| Local paths, benchmark hacks, test bypasses | Met |
| Production parity | Docker image verified; Render runtime NOT met (memory) |
| Production deployment | Verified |
| Real production E2E | **NOT MET** |
| All golden documents passed | **NOT MET** (Radhakrishna in degraded mode; model-dependent accuracy untested) |
| No secrets exposed | **NOT MET** (shared `SESSION_SECRET`) |
