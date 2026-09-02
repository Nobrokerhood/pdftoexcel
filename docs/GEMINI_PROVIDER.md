# Gemini Provider

The project uses the supported `google-genai` package:

```python
from google import genai
from google.genai import types
```

`GeminiDocumentClient` creates a lazy `genai.Client(api_key=...)` and calls:

```python
client.models.generate_content(...)
```

JSON extraction and verification requests use native JSON response mode:

```python
types.GenerateContentConfig(response_mime_type="application/json")
```

Pydantic models remain authoritative after model output:

- `MemberReceiptExtraction`
- `VendorInvoiceExtraction`
- `VerificationResult`
- validation result schemas

Required env vars:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
```

The old `google-generativeai` package is no longer required by this application.
