# Testing Folder Document Analysis

Audit and extraction analysis of real acceptance files in `testing_folder/` for NoBrokerHood (NBH) Accounting AI.

---

## 1. Summary of Test Files

| Filename | Document Type | Page Count | OCR Engine | OCR Lines | Extracted Rows | Target NBH Rows | Primary Bill Heads / Purpose |
|---|---|---|---|---|---|---|---|
| `sample-radhakrishna.pdf` | `SOCIETY_MEMBER_LEDGER` / `MEMBERS_LIST` | 1 (Landscape) | RapidOCR (200 DPI) | 310 lines | 147 member charges | 147 rows | Property Tax, Water Charges, Sinking Fund, Maint. Charges, Electricity Charges |
| `IDFCFIRSTBankstatement_10178263032_134000056 (2)_page-0008.pdf` | `BANK_STATEMENT` | 2 (Portrait) | RapidOCR (200 DPI) | 312 lines | 37 transactions | 37 rows | IMPS, UPI, Cheque, NEFT bank transactions with UTR/Ref |

---

## 2. Detailed Document Analysis

### File 1: `sample-radhakrishna.pdf`
- **Source Document:** Lakshmi Sai Co-op Housing Society Ltd (Members List Period 01/04/2024 to 31/03/2025).
- **Format:** PDF (1 landscape page, 2339 x 1654 rendered pixels).
- **OCR Quality:** High (mean confidence > 0.98, printed tabular layout).
- **Detected Source Columns:** `wing`, `Unit No`, `Member Name`, `Opg.Bal`, `Property Tax`, `Water Charges`, `Sinking Fund`, `Maint. Charges`, `Electricity Charges`, `Cultural / Other Charges`, `Int on Due`, `Deed Of Conveyance`, `Balance`.
- **Accounting Interpretation:** Multi-member periodic billing ledger with row per flat/member and columns per bill head.
- **NBH Target Mapping:**
  - `Payment Type*`: `-`
  - `Society Bank Name/Bank code*`: `-`
  - `Cheque/Ref No*`: `-`
  - `Tower No*`: Wing column (e.g. `A`)
  - `Flat No*`: Unit No (e.g. `1`, `2`, `20`)
  - `Bill Head*`: Column header (e.g. `Property Tax`, `Water Charges`, `Sinking Fund`, `Maint. Charges`, `Electricity Charges`, `Cultural / Other Charges`)
  - `Amount*`: Cell numeric charge value
  - `Transaction Date*`: `-` (or billing period)
  - `Comments`: Member Name (e.g. `Mr. Y. G. Taikar`, `Mr. V. Masurkar`)
  - `Meter No`: `-`
  - `Cheque Issuer Bank`: `-`
  - `Cheque Date`: `-`

### File 2: `IDFCFIRSTBankstatement_10178263032_134000056 (2)_page-0008.pdf`
- **Source Document:** IDFC FIRST Bank Statement of Account (Account 10178263032).
- **Format:** PDF (2 portrait pages, 1653 x 2339 rendered pixels each).
- **OCR Quality:** High (mean confidence > 0.99, standard bank transaction table).
- **Detected Source Columns:** `Transaction Date`, `Value Date`, `Particulars`, `Cheque No`, `Debit`, `Credit`, `Balance`.
- **Accounting Interpretation:** Bank statement containing member payments via IMPS/UPI/NEFT/Cheque, interest credits, and charges across multiple pages.
- **NBH Target Mapping:**
  - `Payment Type*`: Detected payment channel (`IMPS`, `UPI`, `Cheque`, `NEFT`, `Bank`)
  - `Society Bank Name/Bank code*`: Bank code or bank name from description or `-`
  - `Cheque/Ref No*`: UTR / Reference No / Cheque No (e.g. `511625276821`, `300239`)
  - `Tower No*`: `-` (or parsed from narration if present)
  - `Flat No*`: Flat/unit parsed from narration (e.g. `D002`, `B006`, `C001`, `2969`) or `-`
  - `Bill Head*`: Transaction category or purpose in narration (e.g. `PRI Apt APRIL`, `Maintenance`) or `-`
  - `Amount*`: Numeric credit/debit amount (e.g. `14759.00`, `15400.00`)
  - `Transaction Date*`: Date in `DD-MM-YYYY` (e.g. `26-04-2025`, `25-04-2025`)
  - `Comments`: Narration / Beneficiary / Remitter string
  - `Meter No`: `-`
  - `Cheque Issuer Bank`: Bank mentioned in cheque deposit (e.g. `CANARA BANK`) or `-`
  - `Cheque Date`: Date of cheque if present (e.g. `23-04-2025`) or `-`

---

## 3. Reconciled Output Requirements

1. **Standard 12 Target Columns:**
   `Payment Type*` | `Society Bank Name/Bank code(Given to you by nobrokerhood)*` | `Cheque/Ref No*` | `Tower No*` | `Flat No*` | `Bill Head*` | `Amount*` | `Transaction Date*` | `Comments` | `Meter No` | `Cheque Issuer Bank` | `Cheque Date`
2. **Strict Missing Value Formatting:** Every unavailable field is strictly `"-"`.
3. **No Row Loss:** Multi-page bank statements and multi-column ledgers preserve 100% of individual accounting rows without omission.
