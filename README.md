# NoBrokerHood (NBH) Accounting AI

Production-grade automated document accounting engine for NoBrokerHood society and apartment management. Converts unstructured financial documents (bank receipts, vendor bills, petty cash registers, handwritten statements) into validated, schema-compliant Excel imports for the NoBrokerHood ERP system.

---

## 1. Prerequisites

- **Python**: Python 3.11 or 3.12 (64-bit recommended)
- **Poppler Utilities**: Required for PDF rasterization (`pdfinfo`, `pdftoppm`).
  - **Debian / Ubuntu / Docker**: `sudo apt-get install -y poppler-utils libgl1 libglib2.0-0`
  - **macOS**: `brew install poppler`
  - **Windows**: Download Poppler for Windows and set `POPPLER_PATH=C:\path\to\poppler\bin` in `.env` (or add to system `PATH`).
- **Google Cloud / Workspace Access**:
  - Google Cloud OAuth 2.0 Web Client ID for `@nobroker.in` Workspace authentication.
  - Google Service Account with **Content Manager** access to the Shared Drive and **Editor** access to the accounting master spreadsheet.
  - Google Gemini API key (`gemini-2.5-flash`).

---

## 2. Quickstart for Developers (Clean Clone)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nobrokerhood/pdftoexcel.git
cd pdftoexcel
```

### Step 2: Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your development or test credentials:
- `GEMINI_API_KEY`: Your Gemini API key.
- `GOOGLE_CLIENT_ID`: Your Google OAuth 2.0 Client ID.
- `GOOGLE_SERVICE_ACCOUNT_JSON` (or `GOOGLE_SERVICE_ACCOUNT_FILE` pointing to your service account key file).
- `GOOGLE_ACCOUNTING_SPREADSHEET_ID`: Spreadsheet ID for configuration, templates, and logs.
- `GOOGLE_SHARED_DRIVE_ID`: Google Workspace Shared Drive ID.

### Step 5: Start Backend Server
```bash
uvicorn main:app --host 127.0.0.1 --port 8030 --reload
```
Check health:
```bash
curl http://127.0.0.1:8030/health
```

### Step 6: Start Frontend Server
In a separate terminal:
```bash
python -m http.server 5000 --bind 127.0.0.1
```
Open your browser at `http://127.0.0.1:5000/index.html`.

---

## 3. Environment Profiles

| Setting | Local Development | Staging | Production |
|---|---|---|---|
| `ENVIRONMENT` | `development` | `staging` | `production` |
| `DEBUG` | `true` | `false` | `false` |
| `ENABLE_DOCS` | `true` (`/docs` available) | `true` | `false` (gated) |
| `ALLOW_DEV_LOGIN` | `true` or `false` | `false` | `false` (enforced fail-fast) |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5000` | Staging Web Domain | Production Web Domain (No `*`) |
| `Google OAuth Origins` | `http://localhost:5000` | Staging Web Domain | `https://accounting.nobrokerhood.com` |

---

## 4. Running Tests

Run the complete test suite:
```bash
pytest -q
```
Expected: `126 passed, 0 failed, 3 warnings`.

Run specific test modules:
```bash
# Production readiness and security tests
pytest tests/test_production_readiness.py -v

# Handwritten document understanding & reconciliation benchmark
pytest tests/test_handwritten_document_understanding.py -v

# Optional unmapped bill head approval tests
pytest tests/test_unmapped_approval.py -v
```

---

## 5. Docker Deployment

Build and run using the production Dockerfile:
```bash
# Build
docker build -t nbh-accounting-ai:latest .

# Run
docker run -d \
  --name nbh-accounting-ai \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  nbh-accounting-ai:latest
```

Verify health:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/readiness
```

---

## 6. Production Operations & Architecture

For complete enterprise production deployment details, Google Cloud/Workspace configuration steps, Nginx reverse-proxy setup, and incident runbooks, see [`DEPLOYMENT.md`](./DEPLOYMENT.md).
