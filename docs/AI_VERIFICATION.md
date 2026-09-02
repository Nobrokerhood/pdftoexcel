# AI Verification

Verification is independent from extraction.

`VerificationAgent` receives:

- original source bytes
- selected purpose
- template definition
- extracted data

It returns structured field-level checks:

- field
- extracted value
- verified value
- status
- confidence
- evidence
- page number where available

Statuses:

- VERIFIED
- MISMATCH
- NOT_FOUND
- UNCERTAIN

Overall statuses:

- PASSED
- FAILED
- NEEDS_REVIEW

When verification fails, `RepairAgent` may correct only the extraction fields
identified by verification. Retry count is controlled by
`AI_VERIFICATION_MAX_RETRIES`.

Normal tests inject fake verification/repair providers. Live Gemini
verification and repair are available behind provider abstractions and are only
invoked by live processing routes when credentials are configured.
