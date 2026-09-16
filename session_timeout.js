const INACTIVITY_LIMIT = 20 * 60 * 1000;
const HEARTBEAT_INTERVAL = 60 * 1000;
const API_BASE_URL = (typeof window !== "undefined" && window.__API_BASE__ !== undefined)
  ? window.__API_BASE__
  : ((location.hostname === "localhost" || location.hostname === "127.0.0.1")
      ? "http://127.0.0.1:8030"
      : (location.hostname.endsWith("github.io")
          ? "https://pdftoexcel-846x.onrender.com"
          : ""));

let logoutTimer;
let userActive = true;

function resetLogoutTimer() {
  userActive = true;
  clearTimeout(logoutTimer);

  logoutTimer = setTimeout(() => {
    // If a document is currently actively processing, do not log out
    if (window.__ACCOUNTING_JOB_ACTIVE__) {
      resetLogoutTimer();
      return;
    }
    alert("Session expired due to inactivity. Please login again.");
    logoutFromSession();
  }, INACTIVITY_LIMIT);
}

["click", "mousemove", "keypress", "scroll", "touchstart"].forEach(event => {
  document.addEventListener(event, resetLogoutTimer, true);
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    userActive = true;
  }
});

async function heartbeat() {
  const token = sessionStorage.getItem("accounting_session_token");
  if (!token) return;

  const isActive = userActive || !!window.__ACCOUNTING_JOB_ACTIVE__;

  try {
    const response = await fetch(API_BASE_URL + "/auth/heartbeat", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        user_active: isActive,
        page_visible: !document.hidden
      })
    });
    if (response.status === 401) {
      try {
        const verifyRes = await fetch(API_BASE_URL + "/auth/me", {
          headers: { "Authorization": "Bearer " + token }
        });
        if (verifyRes.status === 401) {
          logoutFromSession();
        }
      } catch (err) {
        console.warn("Auth verification network error:", err);
      }
    }
  } catch (error) {
    console.warn("Heartbeat network warning (will retry on next tick):", error);
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
