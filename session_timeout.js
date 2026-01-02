// session_timeout.js
// Auto logout after 20 minutes of inactivity

const INACTIVITY_LIMIT = 20 * 60 * 1000; // 20 minutes
let logoutTimer;

// Reset timer on user activity
function resetLogoutTimer() {
  clearTimeout(logoutTimer);

  logoutTimer = setTimeout(() => {
    alert("Session expired due to inactivity. Please login again.");
    sessionStorage.clear();
    window.location.href = "index.html";
  }, INACTIVITY_LIMIT);
}

// Track user activity
["click", "mousemove", "keypress", "scroll", "touchstart"].forEach(event => {
  document.addEventListener(event, resetLogoutTimer, true);
});

// Start timer when page loads
resetLogoutTimer();
