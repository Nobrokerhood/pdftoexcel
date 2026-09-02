# Template Registry

The registry maps a document purpose to its active output template.

Initial templates:

- `MEMBER_RECEIPT` -> `NBH_MEMBER_RECEIPT_V1`
- `VENDOR_INVOICE` -> `NBH_VENDOR_BILL_V1`

## Member Receipt Fields

- Payment Type
- Society Bank Name/Bank code
- Cheque/Ref No
- Tower No
- Flat No
- Bill Head
- Amount
- Transaction Date
- Comments
- Meter No
- Cheque Issuer Bank
- Cheque Date

## Vendor Invoice Fields

- Bill Number
- Bill Date
- Vendor Code
- Due Date
- Narration
- CGST Amount
- SGST Amount
- IGST Amount
- TDS Amount
- Expense Code
- Expense Amount

Vendor invoice supports multiple expense entries. No extra ERP upload columns
were invented in this milestone.

Generation path:

Approved Canonical Data -> Template Registry -> Template Mapper -> openpyxl ->
XLSX

AI never generates spreadsheet headers.
