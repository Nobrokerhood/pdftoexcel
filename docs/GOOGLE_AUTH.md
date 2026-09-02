# Google Authentication

The frontend receives a Google ID token from Google Sign-In and sends it to
`POST /auth/google-login`.

The backend verifies:

- token signature
- issuer
- audience from `GOOGLE_CLIENT_ID`
- expiration through Google's verifier
- verified email

The browser-supplied email is not trusted for authorization. After verification,
the backend checks `Accounting_AI_User_Master` through `UserMasterService`.

Required for real login:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_USER_MASTER_SHEET_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON` or `GOOGLE_SERVICE_ACCOUNT_FILE`

Optional:

- `ALLOWED_EMAIL_DOMAIN=nobroker.in`
- `ALLOW_DOMAIN_WIDE_ACCESS=false`

Domain-wide access is disabled by default and must not replace User Master
authorization in production.
