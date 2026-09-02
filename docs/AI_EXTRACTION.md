# AI Extraction

Extraction is isolated behind `ExtractionAgent`.

Inputs:

- original source bytes
- selected purpose
- registered template definition

Outputs:

- `MemberReceiptExtraction`
- `VendorInvoiceExtraction`

AI does not decide spreadsheet headers. Template Registry controls the final
output structure.

Normal tests inject fake extraction providers. Live Gemini extraction is behind
`GeminiExtractionProvider` and requires `GEMINI_API_KEY`; backend import does
not require the key. The provider requests JSON output and validates it through
Pydantic schemas before the workflow continues.
