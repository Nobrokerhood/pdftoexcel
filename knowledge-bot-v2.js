// Knowledge Bot v2 - Simplified and robust
console.log('Knowledge Bot v2 loaded');

function initKnowledgeBot() {
  console.log('Initializing Knowledge Bot...');
  
  // Create launcher button
  const launcher = document.createElement('div');
  launcher.id = 'kb-launcher';
  launcher.style.cssText = `
    position: fixed;
    right: 24px;
    bottom: 24px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #00bfff;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    z-index: 2147483647;
    border: none;
  `;
  launcher.innerHTML = '💬';
  launcher.title = 'Open Knowledge Bot';
  
  // Create panel
  const panel = document.createElement('div');
  panel.id = 'kb-panel';
  panel.style.cssText = `
    position: fixed;
    right: 24px;
    bottom: 100px;
    width: 380px;
    max-height: 520px;
    background: linear-gradient(180deg,#041428,#06293a);
    border: 1px solid rgba(0,191,255,0.15);
    border-radius: 12px;
    box-shadow: 0 8px 40px rgba(2,6,23,0.6);
    color: #e6f7ff;
    overflow: hidden;
    display: none;
    flex-direction: column;
    z-index: 99999;
  `;
  
  panel.innerHTML = `
    <div style="padding:12px 14px;border-bottom:1px solid rgba(255,255,255,0.03);display:flex;align-items:center;justify-content:space-between;">
      <strong>Knowledge Bot</strong>
      <div style="font-size:12px;color:#bfe9ff">KB: <span id="kb-name">ERP KB</span></div>
      <button id="kb-close" style="background:none;border:none;color:#e6f7ff;cursor:pointer;font-size:18px;padding:0;width:20px;height:20px;">✕</button>
    </div>
    <div id="kb-messages" style="padding:12px;overflow:auto;height:360px;font-size:14px;line-height:1.4;color:#dff6ff;"></div>
    <div style="padding:10px;border-top:1px solid rgba(255,255,255,0.03);display:flex;gap:8px;">
      <input id="kb-input" placeholder="Ask about the documents..." style="flex:1;padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.05);background:transparent;color:#e6f7ff;outline:none;" />
      <button id="kb-send" style="background:#00bfff;border-radius:8px;padding:8px 10px;border:none;color:#001922;cursor:pointer;font-weight:700">Ask</button>
    </div>
  `;
  
  document.body.appendChild(launcher);
  document.body.appendChild(panel);
  
  // Get elements
  const messagesEl = panel.querySelector('#kb-messages');
  const inputEl = panel.querySelector('#kb-input');
  const sendBtn = panel.querySelector('#kb-send');
  const closeBtn = panel.querySelector('#kb-close');
  
  // Toggle panel
  launcher.addEventListener('click', () => {
    const isVisible = panel.style.display === 'flex';
    panel.style.display = isVisible ? 'none' : 'flex';
    if (!isVisible) inputEl.focus();
  });
  
  closeBtn.addEventListener('click', () => {
    panel.style.display = 'none';
  });
  
  // Load KB data from local file
  let kbData = [];
  
  async function loadKB() {
    try {
      console.log('Loading KB from kb_store.json...');
      const res = await fetch('./kb_store.json');
      if (!res.ok) throw new Error('Failed to load KB');
      
      kbData = await res.json();
      console.log('Loaded KB with', kbData.length, 'documents');
      
      if (Array.isArray(kbData) && kbData.length > 0) {
        appendSystem(`✅ Loaded ${kbData.length} documents. Ask me anything!`);
      } else {
        appendSystem('⚠️ KB is empty or invalid format');
      }
    } catch (err) {
      console.error('Error loading KB:', err);
      appendSystem('⚠️ Could not load KB. Trying backend...');
      // Try to connect to backend
    }
  }
  
  // Append message helpers
  function appendSystem(text) {
    const msg = document.createElement('div');
    msg.style.cssText = 'margin:8px 0;text-align:center;font-size:12px;color:#aeefff;';
    msg.textContent = text;
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  
  function appendUser(text) {
    const msg = document.createElement('div');
    msg.style.cssText = 'margin:8px 0;text-align:right;';
    msg.innerHTML = `<div style="display:inline-block;background:rgba(0,191,255,0.12);padding:8px 10px;border-radius:10px;color:#bff7ff">${escapeHtml(text)}</div>`;
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  
  function appendBot(text) {
    const msg = document.createElement('div');
    msg.style.cssText = 'margin:8px 0;text-align:left;';
    msg.innerHTML = `<div style="display:inline-block;background:rgba(255,255,255,0.03);padding:8px 10px;border-radius:10px;color:#e6f7ff">${escapeHtml(text)}</div>`;
    messagesEl.appendChild(msg);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
  
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[c]);
  }
  
  // Search in KB
  function searchKB(query) {
    if (kbData.length === 0) return null;
    
    const q = query.toLowerCase();
    for (const doc of kbData) {
      const text = (doc.text || '').toLowerCase();
      if (text.includes(q)) {
        return {
          source: doc.source || 'Document',
          excerpt: (doc.text || '').substring(0, 300)
        };
      }
    }
    return null;
  }
  
  // Ask function
  async function ask() {
    const query = inputEl.value.trim();
    if (!query) return;
    
    appendUser(query);
    inputEl.value = '';
    
    if (kbData.length === 0) {
      appendBot('Loading knowledge base...');
      await loadKB();
      return;
    }
    
    // Search locally
    const result = searchKB(query);
    if (result) {
      appendBot(`📄 From ${result.source}:\n\n${result.excerpt}...`);
    } else {
      appendBot('I could not find matching information. Try different keywords or check the Load sample KB button.');
    }
  }
  
  // Event listeners
  sendBtn.addEventListener('click', ask);
  inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') ask();
  });
  
  // Load KB on initialization
  loadKB();
  
  console.log('Knowledge Bot initialized successfully');
}

// Auto-initialize when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initKnowledgeBot);
} else {
  initKnowledgeBot();
}
