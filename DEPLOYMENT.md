# NoBrokerHood (NBH) Accounting AI - Production Deployment Guide

## 1. System Requirements & Architecture
- **Runtime**: Python 3.11 or 3.12 (Debian/Ubuntu Linux or Docker recommended for production).
- **Core Server**: FastAPI running via Uvicorn (ASGI).
- **Document Preprocessing**:
  - `poppler-utils` (`pdftoppm`, `pdfinfo`) for rendering PDF page images.
  - `rapidocr-onnxruntime` + `onnxruntime` for CPU-based spatial OCR.
- **AI Intelligence**: Google Gemini 2.5 Flash via `google-genai` SDK.
- **Enterprise Storage**: Google Workspace Shared Drive (`NoBrokerHood Accounting AI`).
- **Configuration & Audit**: Google Sheets spreadsheet (`Accounting_AI_Config_and_Audit`).

---

## 2. Environment Variables Specification
Configure these environment variables in your deployment platform (Render, AWS ECS, GCP Cloud Run, or Kubernetes Secret). Never commit these values to source control.

| Variable Name | Required | Example / Format | Description |
|---|---|---|---|
| `ENVIRONMENT` | Yes | `production` | Enables production security checks and disables dev tooling. |
| `DEBUG` | Yes | `false` | Disables verbose debug logging and stack trace leakage. |
| `LOG_LEVEL` | Yes | `INFO` | Configures logging verbosity (`INFO`, `WARNING`, `ERROR`). |
| `ENABLE_DOCS` | Yes | `false` | Set to `false` to disable `/docs`, `/redoc`, and `/openapi.json`. |
| `ALLOW_DEV_LOGIN` | Yes | `false` | Must be `false` in production. Disables `/auth/dev-login`. |
| `GEMINI_API_KEY` | Yes | `AIzaSy...` | API key from Google AI Studio / Google Cloud. |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Model code used for extraction and verification. |
| `GOOGLE_CLIENT_ID` | Yes | `...apps.googleusercontent.com` | Google Cloud OAuth 2.0 Web Client ID. |
| `ALLOWED_EMAIL_DOMAIN` | Yes | `nobroker.in` | Restricts logins to corporate Google Workspace accounts. |
| `ALLOW_DOMAIN_WIDE_ACCESS` | No | `true` | Allows domain users access without individual `User_Master` entries. |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Yes | `{"type": "service_account", ...}` | Complete JSON key string for service account. |
| `GOOGLE_ACCOUNTING_SPREADSHEET_ID` | Yes | `1BxiMVs0XR...` | Spreadsheet ID containing folder configs, audit, and job state. |
| `GOOGLE_SHARED_DRIVE_ID` | Yes | `0AFMzLN8WYbebUk9PVA` | Google Workspace Shared Drive ID. |
| `CORS_ALLOWED_ORIGINS` | Yes | `https://accounting.nobrokerhood.com` | Comma-separated list of allowed frontend origins (no `*`). |
| `SESSION_INACTIVITY_SECONDS` | No | `1200` | Session timeout in seconds (default: 20 minutes). |
| `MAX_FILE_SIZE_MB` | No | `10` | Maximum uploaded PDF/image file size in MB. |
| `POPPLER_PATH` | No | *(leave empty on Linux)* | Filesystem path if `pdftoppm` is not in system `$PATH`. |

---

## 3. Google Cloud / Workspace Configuration

This section details all external configurations required in Google Cloud Console and Google Workspace Admin Console before deployment.

### A. Google Cloud APIs
Ensure the following APIs are enabled in your Google Cloud Project:
1. **Google Drive API** (`drive.googleapis.com`)
2. **Google Sheets API** (`sheets.googleapis.com`)
3. **Generative Language API** (`generativelanguage.googleapis.com`) for Gemini 2.5 Flash

### B. Production Google OAuth Setup
Follow these steps to configure Google OAuth for production:
1. **Create/use Google OAuth Web Client**: In Google Cloud Console -> APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> Application type: **Web application**.
2. **Copy OAuth Web Client ID**: It has the format `...apps.googleusercontent.com`.
3. **Configure GOOGLE_CLIENT_ID on production server**: In Render Dashboard (or production environment): `Render Dashboard -> Service (pdftoexcel) -> Environment -> Add Environment Variable` -> Key: `GOOGLE_CLIENT_ID`, Value: `<OAuth Web Client ID>`.
4. **Configure production frontend origin**: Under Authorized JavaScript Origins in Google Cloud Console, add:
   - `https://nobrokerhood.github.io`
   - `https://pdftoexcel-846x.onrender.com`
   - Production domain (e.g., `https://accounting.nobrokerhood.com`)
   - `http://localhost:5000` / `http://127.0.0.1:5000`
5. **Configure redirect URI only if required**: None required. The frontend uses **Google Identity Services (GIS)** ID-token flow (`window.google.accounts.id.initialize`), receiving tokens via JavaScript callback without redirecting.
6. **Configure Workspace access**: In Google Workspace Admin Console, ensure the OAuth app is authorized/internal for the organization.
7. **Configure ALLOWED_EMAIL_DOMAIN=nobroker.in**: Set this environment variable so that only `@nobroker.in` Workspace identities can authenticate. Set `ALLOW_DOMAIN_WIDE_ACCESS=true` to permit all organization employees without individual pre-registration.
8. **Deploy/restart backend**: Ensure backend service restarts with the new environment variables.
9. **Verify /config/public**: Test `GET https://<api-url>/config/public` to ensure `google_client_id` is populated and no secrets are exposed.
10. **Test Google login**: Open the production login page, click **Continue with Google**, authenticate with an authorized `@nobroker.in` account, and confirm access.

> [!CAUTION]
> **CRITICAL CREDENTIAL DISTINCTION**:
> The service-account `client_id` is NOT the employee Google OAuth client ID.
> 
> The application uses three completely separate Google identity mechanisms:
> - **Employee Google OAuth (`GOOGLE_CLIENT_ID`)**: Google Cloud OAuth 2.0 Web Client ID used strictly for employee browser login.
> - **Backend Service Account (`GOOGLE_SERVICE_ACCOUNT_JSON` / `FILE`)**: Service Account JSON credentials used strictly for automated backend Google Drive and Google Sheets storage operations.
> - **Gemini AI SDK (`GEMINI_API_KEY`)**: Google AI Studio / Vertex AI API key used strictly for multimodal financial document extraction and verification.
> 
> Never pass the service account client ID or credentials to `GOOGLE_CLIENT_ID` or frontend OAuth.

### C. Google Workspace User Authorization Model
- The application enforces domain restriction at the backend level via `ALLOWED_EMAIL_DOMAIN=nobroker.in`.
- When `ALLOW_DOMAIN_WIDE_ACCESS=true`, **any verified `@nobroker.in` employee** can log in and process documents as `role="USER"` without needing their email pre-entered in the `User_Master` sheet.
- If specific users require administrative privileges (`role="ADMIN"`), add their row in the `User_Master` tab of the configured accounting spreadsheet.

### D. Service Account & Shared Drive Permissions
1. **Service Account Creation**:
   - Create a dedicated Service Account in Google Cloud (e.g., `nbh-accounting-worker@<project-id>.iam.gserviceaccount.com`).
   - Create and download a JSON key. Provide it to the application via `GOOGLE_SERVICE_ACCOUNT_JSON` or `GOOGLE_SERVICE_ACCOUNT_FILE`.
2. **Google Workspace Shared Drive (`GOOGLE_SHARED_DRIVE_ID`)**:
   - The Service Account email must be added directly as a member of the Shared Drive with the **Content Manager** role (or **Manager**).
   - This grants permissions to create, move, list, and read files in the purpose-specific folders (`Incoming`, `Review`, `Completed`, `Output`).
   - All backend Drive calls use `supportsAllDrives=True` to interact with Shared Drive items seamlessly.
3. **Google Sheets Configuration (`GOOGLE_ACCOUNTING_SPREADSHEET_ID`)**:
   - Share the master spreadsheet (`Accounting_AI_Config_and_Audit`) with the Service Account email as **Editor**.
   - Required worksheet tabs: `User_Master`, `Folder_Config`, `Template_Master`, `Mapping_Master`, `Processing_Log`, `Activity_Log`, `Login_Audit`, `Session_Log`, `API_Usage_Report`.

### E. Google Gemini AI API Configuration
- Generate an API key for Google Gemini from Google Cloud Console or Google AI Studio.
- Configure `GEMINI_API_KEY=<key>` and `GEMINI_MODEL=gemini-2.5-flash`.
- No fallback key or multi-key logic is required.

---

## 4. Container Deployment (Docker)
Build and run using the optimized production Dockerfile:

```bash
# 1. Build image
docker build -t nbh-accounting-ai:latest .

# 2. Run container with environment file
docker run -d \
  --name nbh-accounting-ai \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env.production \
  nbh-accounting-ai:latest
```

The container runs as a non-root user (`appuser`), includes `poppler-utils`, and listens on port 8000.

---

## 5. Non-Docker Linux Host Setup
If deploying directly on an Ubuntu/Debian virtual machine:

```bash
# 1. Install system dependencies
sudo apt-get update && sudo apt-get install -y poppler-utils libgl1 libglib2.0-0

# 2. Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Start production ASGI server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 6. Reverse Proxy & HTTPS Configuration (Nginx Example)
Always terminate SSL/TLS at the reverse proxy (or cloud load balancer) to ensure all traffic uses HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name accounting.nobrokerhood.com;

    ssl_certificate /etc/letsencrypt/live/accounting.nobrokerhood.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/accounting.nobrokerhood.com/privkey.pem;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Static UI Files (served directly)
    location / {
        root /var/www/nbh-accounting-ai;
        index index.html;
        try_files $uri $uri/ =404;
    }

    # Backend API Routing
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        client_max_body_size 12M;
    }
}
```

---

## 7. Health & Monitoring Probes
- **Liveness Probe**: `GET /health`
  - Returns `200 OK`: `{"status": "healthy", "service": "nbh-accounting-ai"}`
- **Readiness Probe**: `GET /readiness`
  - Returns `200 OK`:
    ```json
    {
      "status": "ready",
      "checks": {
        "gemini_configured": true,
        "google_credentials_configured": true,
        "poppler_available": true,
        "rapidocr_ready": true
      }
    }
    ```

---

## 8. Staging Validation Steps
Before switching production DNS:
1. Deploy to staging environment with real credentials but pointing to staging Drive folders.
2. Verify Google Workspace login with a corporate `@nobroker.in` email.
3. Upload `WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf`:
   - Confirm 29 transactions appear in Human Review.
   - Confirm Row 21 is ₹2,500, Row 23 is ₹1,789, Row 28 is ₹750.
   - Confirm Inflows ($₹10,000, ₹68,800, ₹30,000$) remain outside the grid.
   - Confirm Opening Balance distinguishes Source ($₹1,714$) from Derived ($₹2,194$).
   - Confirm Closing Balance displays Source ($₹10,174$) vs Calculated ($₹10,177$) with $₹3$ discrepancy.
   - Click **Approve & Generate Excel** without confirming mappings.
   - Confirm Excel downloads with exact 12 NBH columns and unmapped bill heads preserved.

---

## 9. Rollback Procedure
If an issue arises post-deployment:
1. **Container Rollback**:
   ```bash
   docker stop nbh-accounting-ai
   docker run -d --name nbh-accounting-ai -p 8000:8000 --env-file .env.production <PREVIOUS_IMAGE_TAG>
   ```
2. **Database / Sheets Safety**:
   - Google Drive and Google Sheets do not execute destructive schema migrations. All data remains stored in the Shared Drive and Sheets unchanged.
