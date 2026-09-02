# Session Tracking

`POST /auth/google-login` creates an application session after Google token
verification and User Master authorization.

Session fields:

- session_id
- email
- name
- role
- login_at
- last_seen_at
- logout_at
- active_duration_seconds
- status

The session token is a random server-side bearer token stored in browser
`sessionStorage`. Google passwords and Google ID tokens are not stored in
Sheets.

Heartbeat endpoint:

- `POST /auth/heartbeat`

Heartbeat body:

- `user_active`
- `page_visible`

Active time is increased only when the page reports activity and visibility.
The increment is capped by `SESSION_HEARTBEAT_GRACE_SECONDS`, so active time is
not simply `logout - login`.

Environment:

- `SESSION_INACTIVITY_SECONDS=1200`
- `SESSION_HEARTBEAT_GRACE_SECONDS=120`
