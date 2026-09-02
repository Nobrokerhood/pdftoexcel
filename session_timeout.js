const INACTIVITY_LIMIT = 20 * 60 * 1000;
const HEARTBEAT_INTERVAL = 60 * 1000;
const API_BASE_URL = location.hostname === "localhost" || location.hostname === "127.0.0.1"
  ? "http://127.0.0.1:8030"
  : "https://pdftoexcel-846x.onrender.com";

let logoutTimer;
let userActive = true;

function resetLogoutTimer() {
  userActive = true;
  clearTimeout(logoutTimer);

  logoutTimer = setTimeout(() => {
    alert("Session expired due to inactivity. Please login again.");
    logoutFromSession();
  }, INACTIVITY_LIMIT);
}

["click", "mousemove", "keypress", "scroll", "touchstart"].forEach(event => {
  document.addEventListener(event, resetLogoutTimer, true);
});

document.addEventListener("visibilitychange", () => {
  userActive = !document.hidden;
});

async function heartbeat() {
  const token = sessionStorage.getItem("accounting_session_token");
  if (!token) return;

  try {
    const response = await fetch(API_BASE_URL + "/auth/heartbeat", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        user_active: userActive,
        page_visible: !document.hidden
      })
    });
    if (response.status === 401) {
      logoutFromSession();
    }
  } catch (error) {
    console.warn("Heartbeat failed.");
  } finally {
    userActive = false;
  }
}

async function logoutFromSession() {
  const token = sessionStorage.getItem("accounting_session_token");
  if (token) {
    await fetch(API_BASE_URL + "/auth/logout", {
      method: "POST",
      headers: { "Authorization": "Bearer " + token }
    }).catch(() => {});
  }
  sessionStorage.clear();
  window.location.href = "index.html";
}

resetLogoutTimer();
setInterval(heartbeat, HEARTBEAT_INTERVAL);
