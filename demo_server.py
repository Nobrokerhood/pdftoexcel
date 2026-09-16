"""Standalone sample-data demo. Run: python demo_server.py"""
import io
import re
from pathlib import Path
from fastapi import Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image, ImageDraw
from app.app_factory import create_app
from demo_services import (settings, FakeSheetsService, FakeDriveService, FakeVerifier,
    StaticExtractionProvider, StaticVerificationProvider, StaticRepairProvider,
    records, MEMBER_DATA, VENDOR_DATA)

ROOT = Path(__file__).resolve().parent
app = create_app(settings=settings(), sheets_service=FakeSheetsService(records()),
    drive_service=FakeDriveService(), google_token_verifier=FakeVerifier(),
    extraction_provider=StaticExtractionProvider({'MEMBER_RECEIPT': MEMBER_DATA, 'VENDOR_INVOICE': VENDOR_DATA}),
    verification_provider=StaticVerificationProvider(), repair_provider=StaticRepairProvider(MEMBER_DATA))
app.router.routes = [
    r for r in app.router.routes
    if getattr(r, "path", None) not in {"/", "/index.html", "/accounting.html", "/session_timeout.js"}
]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

@app.middleware('http')
async def local_only(request: Request, call_next):
    if request.client and request.client.host not in {'127.0.0.1', '::1', 'testclient'}:
        return JSONResponse({'detail':'Local demo only'}, status_code=403)
    origin = request.headers.get('origin')
    if request.method not in {'GET','HEAD','OPTIONS'} and origin and origin not in {'http://127.0.0.1:8031','http://localhost:8031'}:
        return JSONResponse({'detail':'Use the local demo page'}, status_code=403)
    if request.url.path.startswith(('/kb', '/process-document', '/export-to-excel')):
        return JSONResponse({'detail':'This feature requires live services. Use the sample accounting workflow.'},status_code=503)
    return await call_next(request)

@app.post('/demo/login')
def demo_login():
    user = app.state.user_master_service.authorize('demo@example.test')
    session = app.state.session_service.create_session(user)
    return {**session.public_dict(), 'session_token':session.token}

@app.get('/', response_class=HTMLResponse)
@app.get('/index.html', response_class=HTMLResponse)
def entry():
    return '''<!doctype html><html><head><title>Accounting AI | Local demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"><style>
    body{font:17px system-ui;background:#f6f8fb;color:#172033;display:grid;place-items:center;min-height:95vh}
    main{max-width:520px;background:white;padding:40px;border:1px solid #d9e1ec;border-radius:12px}
    button{background:#0f766e;color:white;border:0;padding:14px 24px;border-radius:6px;font:inherit;cursor:pointer}
    </style></head><body><main><p>LOCAL DEMO · SAMPLE DATA</p><h1>Explore Accounting AI</h1>
    <p>No Google account is needed. Try a sample receipt or invoice, edit its fields, approve it and download Excel.</p>
    <p>AI extraction and verification are simulated. Files and changes stay in this demo's memory and reset when its server restarts.</p>
    <button id="enter">Enter local demo</button><p id="status" role="status"></p></main><script>
    document.getElementById('enter').onclick=async()=>{try{
    const r=await fetch('/demo/login',{method:'POST'});if(!r.ok)throw Error('Demo login failed');const d=await r.json();
    sessionStorage.setItem('accounting_session_token',d.session_token);
    sessionStorage.setItem('accounting_session_id',d.session_id);
    sessionStorage.setItem('nobroker_user',d.email);
    sessionStorage.setItem('accounting_user',JSON.stringify(d));location.href='/accounting.html';
    }catch(e){document.getElementById('status').textContent=e.message;}};
    </script></body></html>'''

@app.get('/demo/sample/{purpose}')
def sample(purpose: str):
    if purpose not in {'MEMBER_RECEIPT','VENDOR_INVOICE'}:
        return Response(status_code=404)
    data=MEMBER_DATA if purpose=='MEMBER_RECEIPT' else VENDOR_DATA
    im=Image.new('RGB',(950,750),'white'); draw=ImageDraw.Draw(im)
    draw.text((35,25),'SAMPLE DOCUMENT - LOCAL DEMO - NOT A REAL TRANSACTION',fill='black',font_size=23)
    y=80
    for key,value in data.items():
        if value is not None:
            text=f'{key}: {value}'
            for start in range(0,len(text),80):
                draw.text((35,y),text[start:start+80],fill='black',font_size=18);y+=30
    out=io.BytesIO();im.save(out,format='PNG')
    return Response(out.getvalue(),media_type='image/png')

@app.get('/accounting.html', response_class=HTMLResponse)
def accounting():
    html=(ROOT/'accounting.html').read_text(encoding='utf-8')
    html=re.sub(r'const BASE_URL = .*?;', 'const BASE_URL = location.origin;',html,count=1,flags=re.S)
    html=html.replace('<body>', '''<body><div style="padding:16px;background:#fff3cd;color:#553c00;border-bottom:1px solid #e1c569">
    <strong>LOCAL DEMO · SAMPLE DATA</strong> — Extraction always returns sample values, even for uploaded files. AI checks and cloud storage are simulated. Data resets on restart.
    <button id="sampleBtn" type="button" onclick="loadDemoSample()">Load sample for selected purpose</button></div>''')
    html=html.replace('</body>', '''<script>
    async function loadDemoSample(){
      const btn=document.getElementById('sampleBtn');btn.disabled=true;
      try{const purpose=document.getElementById('purpose').value;
      const r=await fetch('/demo/sample/'+purpose);if(!r.ok)throw Error('Sample unavailable');
      const file=new File([await r.blob()],'DEMO-'+purpose+'.png',{type:'image/png'});
      const transfer=new DataTransfer();transfer.items.add(file);
      document.getElementById('file').files=transfer.files;showPreview();await startProcessing();
      }catch(e){setStatus(e.message);}finally{btn.disabled=false;}
    }
    </script></body>''')
    html=re.sub(r'<a[^>]+href="(?:ocr|voice)\.html[^"]*"[^>]*>.*?</a>', '', html, flags=re.S)
    return html

@app.get('/session_timeout.js')
def session_script():
    js=(ROOT/'session_timeout.js').read_text(encoding='utf-8')
    js=re.sub(r'const API_BASE_URL = .*?;', 'const API_BASE_URL = location.origin;',js,count=1,flags=re.S)
    return Response(js,media_type='application/javascript')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8031)
