main.py to run this use the below code line<br>
**uvicorn main:app --reload --port 8000**
<img width="987" height="285" alt="image" src="https://github.com/user-attachments/assets/dea7c5f1-496e-402c-996d-a4c098cbdfac" />


open and and access the html and to work <br>
**python3 -m http.server 5000**<br>
**http://localhost:5000/index.html**
<img width="987" height="285" alt="image" src="https://github.com/user-attachments/assets/ae16b89d-214d-4693-aea3-8c9305d587c2" />

befor deploying the code to rendar update the basic_URL OCR.HTML file current code "const BASE_URL = "http://localhost:8000";<br>
delete the function **# ------------------- CORS for Local Dev -------------------**

```
#**use this code to run this code on local Below**

#------------------- CORS for Local Dev -------------------

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#Add / update CORS settings for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",   # where you serve ocr.html (your screenshots)
        "http://127.0.0.1:5000",
        "http://localhost:8000",   # backend origin (if frontend served same origin)
        "http://127.0.0.1:8000",
        "http://localhost:5500",   # common dev servers
        "http://127.0.0.1:5500",
        # "https://nobrokerhood.github.io",   # keep production if needed
        # "https://nobrokerhood.github.io/pdftoexcel",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
===========================================================================================================
You will get the login audit and app usage report, new code update <br>
# file name :- API_Usage_Report <br>
<img width="987" height="65" alt="image" src="https://github.com/user-attachments/assets/e8d7199e-ae34-4ef4-b13b-f26a97217115" />


# file name:- Login_Audit_Report
<img width="987" height="65" alt="image" src="https://github.com/user-attachments/assets/97ad82ec-079a-44a8-a84c-1578a876ed29" />





# **Login Page**

<img width="1357" height="587" alt="image" src="https://github.com/user-attachments/assets/ca3689a2-348d-4a07-b62a-0bc313cf0378" />

# **Docs OCR Page**

<img width="1357" height="587" alt="image" src="https://github.com/user-attachments/assets/cc890423-9d24-41eb-aeb7-ab772a13bae7" />

# **🎙️ Voice to Excel Entry**

<img width="1357" height="587" alt="image" src="https://github.com/user-attachments/assets/1e72a741-ccd6-4edd-9e25-4d1df3763263" />

## Accounting AI Workflow

New local backend:

`uvicorn main:app --reload --port 8030`

New frontend:

`python -m http.server 5000`

Open:

`http://127.0.0.1:5000/index.html`

The new `accounting.html` page supports the first workflow shell for:

- `MEMBER_RECEIPT`
- `VENDOR_INVOICE`

The flow creates a processing job, uploads the original source to Drive,
runs the LangGraph workflow to mandatory human review, allows edits/mapping
resolution, and generates XLSX only after approval.

## Live Google Configuration

Use this `.env` file:

`C:\Users\virub\Desktop\genai\AI_Agent\legacy_repos\pdftoexcel\.env`

Required live variables:

```env
GOOGLE_CLIENT_ID=
VITE_GOOGLE_CLIENT_ID=
ALLOWED_EMAIL_DOMAIN=nobroker.in
GOOGLE_SERVICE_ACCOUNT_FILE=
GOOGLE_SERVICE_ACCOUNT_JSON=
GOOGLE_ACCOUNTING_SPREADSHEET_ID=
GOOGLE_DRIVE_ROOT_FOLDER_ID=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

Optional legacy per-sheet overrides take priority for their individual tables:

```env
GOOGLE_USER_MASTER_SHEET_ID=
GOOGLE_LOGIN_AUDIT_SHEET_ID=
GOOGLE_SESSION_LOG_SHEET_ID=
GOOGLE_ACTIVITY_LOG_SHEET_ID=
GOOGLE_PROCESSING_LOG_SHEET_ID=
GOOGLE_TEMPLATE_MASTER_SHEET_ID=
GOOGLE_FOLDER_CONFIG_SHEET_ID=
GOOGLE_MAPPING_MASTER_SHEET_ID=
GOOGLE_API_USAGE_SHEET_ID=
```

Shared spreadsheet tabs are documented in `docs/LIVE_GOOGLE_SETUP.md`.
Run `python tools/bootstrap_google_resources.py --check` before live use.


