main.py to run this use the below code line<br>
**uvicorn main:app --reload --port 8000**
<img width="987" height="285" alt="image" src="https://github.com/user-attachments/assets/dea7c5f1-496e-402c-996d-a4c098cbdfac" />


open and and access the html and to work <br>
**python3 -m http.server 5000**<br>
**http://localhost:5000/index.html**
<img width="987" height="285" alt="image" src="https://github.com/user-attachments/assets/ae16b89d-214d-4693-aea3-8c9305d587c2" />

befor deploying the code to rendar update the basic_URL OCR.HTML file current code "const BASE_URL = "http://localhost:8000";<br>
delete the function **# ------------------- CORS for Local Dev -------------------**


#**use this code to run this code on local Below**

# ------------------- CORS for Local Dev -------------------

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add / update CORS settings for local development
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

===========================================================================================================
you will get the login audit and app usage report new code update <br>
file name :- API_Usage_Report <br>

file name :- Login_Audit_Report





# **Login Page**

<img width="1357" height="587" alt="image" src="https://github.com/user-attachments/assets/ca3689a2-348d-4a07-b62a-0bc313cf0378" />

#**Docs OCR Page**

<img width="1357" height="587" alt="image" src="https://github.com/user-attachments/assets/cc890423-9d24-41eb-aeb7-ab772a13bae7" />

#**🎙️ Voice to Excel Entry**

<img width="1357" height="587" alt="image" src="https://github.com/user-attachments/assets/1e72a741-ccd6-4edd-9e25-4d1df3763263" />


