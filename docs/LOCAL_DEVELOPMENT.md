# Local Development

Work only in:

`C:\Users\virub\Desktop\genai\AI_Agent\legacy_repos\pdftoexcel`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run backend:

```powershell
uvicorn main:app --reload --port 8030
```

Run frontend:

```powershell
python -m http.server 5000
```

Open:

`http://127.0.0.1:5000/index.html`

Useful checks:

```powershell
python -m pytest -q
python -m compileall main.py app tests kb tools
node --check session_timeout.js
node --check knowledge-bot-v2.js
python tools/bootstrap_google_resources.py --check --mock
```

Admin users can open the compact Configuration Health panel from `accounting.html`.
