import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* page background: dark navy with a soft lime glow in the corner */
.stApp{background:radial-gradient(900px 500px at 85% -10%, rgba(182,255,60,.10), transparent 60%), #0B0E14}
html, body, [class*="css"]{font-family:'Space Grotesk',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.5px}

/* tidy Streamlit chrome (keep the sidebar open/close button!) */
[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"], footer{display:none}
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"]{
  visibility:visible !important}

/* sidebar: links light up on hover, and the page you are on is highlighted */
[data-testid="stSidebar"]{background:#0F1320;border-right:1px solid #232B3F}
[data-testid="stSidebar"] a{border-radius:10px;transition:background .15s ease, box-shadow .15s ease}
[data-testid="stSidebar"] a:hover{background:#1B2236 !important}
[data-testid="stSidebar"] a[aria-current="page"],
[data-testid="stSidebar"] [aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"]{
  background:rgba(182,255,60,.14) !important;box-shadow:inset 3px 0 0 #B6FF3C !important;border-radius:10px}
[data-testid="stSidebar"] a[aria-current="page"] *{color:#FFFFFF !important;font-weight:600}
[data-testid="stPageLink"] a{border-radius:10px;transition:background .15s ease}
[data-testid="stPageLink"] a:hover{background:#1B2236}

/* numbers look techy */
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;font-weight:600}
[data-testid="stMetricLabel"]{color:#8A94AD;text-transform:uppercase;letter-spacing:1px;font-size:.75rem}

/* bordered cards glow and lift on hover */
[data-testid="stVerticalBlockBorderWrapper"]{background:#131826;border-color:#232B3F;border-radius:14px;
  transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease}
[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-3px);border-color:#B6FF3C;
  box-shadow:0 8px 30px rgba(182,255,60,.12)}

/* hero banner */
.hero{background:linear-gradient(135deg,#131826,#0F1320);border:1px solid #232B3F;border-radius:16px;
  padding:44px;margin-bottom:12px;box-shadow:0 0 70px rgba(182,255,60,.07)}
.hero .eyebrow{color:#B6FF3C;font-family:'JetBrains Mono',monospace;letter-spacing:2px;text-transform:uppercase;font-size:.8rem;display:flex;gap:14px;align-items:center}
.hero h1{font-size:4rem;margin:.4rem 0;padding:0;background:linear-gradient(90deg,#fff 30%,#B6FF3C);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:#8A94AD;font-size:1.2rem;max-width:560px;margin:0}

/* status pill with a pulsing dot */
.pill{display:inline-flex;align-items:center;gap:8px;border-radius:99px;padding:3px 12px;font-size:.78rem;font-weight:600;letter-spacing:.5px}
.pill i{width:8px;height:8px;border-radius:50%;display:inline-block;animation:pulse 1.6s infinite}
.pill.ok{background:rgba(182,255,60,.12);color:#B6FF3C}.pill.ok i{background:#B6FF3C}
.pill.warn{background:rgba(255,184,77,.12);color:#FFB84D}.pill.warn i{background:#FFB84D}
.pill.bad{background:rgba(255,107,122,.12);color:#FF6B7A}.pill.bad i{background:#FF6B7A}
@keyframes pulse{0%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}100%{opacity:1;transform:scale(1)}}
"""


def inject() -> None:
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def status_pill(status) -> str:
    """HTML for a coloured status pill: healthy = lime, degraded = amber, anything else = red."""
    s = str(status).lower()
    kind = "ok" if "healthy" in s or s == "ok" else "warn" if "degrad" in s or "warn" in s else "bad"
    return f'<span class="pill {kind}"><i></i>{str(status).upper()}</span>'