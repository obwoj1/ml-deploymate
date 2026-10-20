import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* page background: dark navy with a soft lime glow in the corner */
.stApp{background:radial-gradient(900px 500px at 85% -10%, rgba(182,255,60,.10), transparent 60%), #0B0E14}
html, body, [class*="css"]{font-family:'Space Grotesk',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.5px}

/* pages fade in */
.block-container{animation:fadein .4s ease}
@keyframes fadein{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

/* tidy Streamlit chrome (keep the sidebar open/close button!) */
[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"], footer{display:none}
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"]{
  visibility:visible !important}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *{color:#B6FF3C !important}

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
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:1.9rem}
[data-testid="stMetricValue"] > div{overflow:visible !important;text-overflow:clip !important;white-space:normal !important}
[data-testid="stMetricLabel"]{color:#8A94AD;text-transform:uppercase;letter-spacing:1px;font-size:.75rem}

/* bordered cards glow and lift on hover */
[data-testid="stVerticalBlockBorderWrapper"]{background:#131826;border-color:#232B3F;border-radius:14px;
  transition:transform .15s ease, border-color .15s ease, box-shadow .15s ease}
[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-3px);border-color:#B6FF3C;
  box-shadow:0 8px 30px rgba(182,255,60,.12)}

.num{font-family:'JetBrains Mono',monospace;color:#B6FF3C;font-size:.85rem;letter-spacing:2px}
.ico{font-size:1.6rem;margin:6px 0}

/* hero banner */
.hero{background-image:linear-gradient(rgba(255,255,255,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,.03) 1px,transparent 1px),linear-gradient(135deg,#131826,#0F1320);
  background-size:36px 36px,36px 36px,auto;
  border:1px solid #232B3F;border-radius:16px;padding:44px;margin-bottom:12px;box-shadow:0 0 70px rgba(182,255,60,.07)}
.hero .eyebrow{color:#B6FF3C;font-family:'JetBrains Mono',monospace;letter-spacing:2px;text-transform:uppercase;font-size:.8rem;display:flex;gap:14px;align-items:center}
.hero h1{font-size:4rem;margin:.4rem 0;padding:0;background:linear-gradient(90deg,#fff 30%,#B6FF3C);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{color:#8A94AD;font-size:1.2rem;max-width:560px;margin:0}


/* coloured stat cards */
.stat{background:#131826;border:1px solid #232B3F;border-top:3px solid var(--c,#B6FF3C);border-radius:14px;padding:18px 20px}
.stat .k{font:12px 'JetBrains Mono',monospace;color:#8A94AD;text-transform:uppercase;letter-spacing:1px}
.stat .v{font:600 1.7rem 'JetBrains Mono',monospace;margin-top:6px;color:var(--c,#E8ECF5)}

/* status pill with a pulsing dot */
.pill{display:inline-flex;align-items:center;gap:8px;border-radius:99px;padding:3px 12px;font-size:.78rem;font-weight:600;letter-spacing:.5px}
.pill i{width:8px;height:8px;border-radius:50%;display:inline-block;animation:pulse 1.6s infinite}
.pill.ok{background:rgba(182,255,60,.12);color:#B6FF3C}.pill.ok i{background:#B6FF3C}
.pill.warn{background:rgba(255,184,77,.12);color:#FFB84D}.pill.warn i{background:#FFB84D}
.pill.bad{background:rgba(255,107,122,.12);color:#FF6B7A}.pill.bad i{background:#FF6B7A}
@keyframes pulse{0%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}100%{opacity:1;transform:scale(1)}}


/* pulsing red alert banner */
.alert-banner{background:rgba(255,107,122,.12);border:1px solid #FF6B7A;color:#FFB3BB;border-radius:12px;
  padding:14px 18px;font-weight:600;animation:glow 2s infinite}
@keyframes glow{0%,100%{box-shadow:0 0 0 rgba(255,107,122,0)}50%{box-shadow:0 0 24px rgba(255,107,122,.35)}}

"""


def inject() -> None:
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def status_pill(status) -> str:
    """HTML for a coloured status pill: healthy = lime, degraded = amber, anything else = red."""
    s = str(status).lower()
    kind = "ok" if "healthy" in s or s == "ok" else "warn" if "degrad" in s or "warn" in s else "bad"
    return f'<span class="pill {kind}"><i></i>{str(status).upper()}</span>'



def status_color(status) -> str:
    s = str(status).lower()
    return "#B6FF3C" if "healthy" in s or s == "ok" else "#FFB84D" if "degrad" in s or "warn" in s else "#FF6B7A"


def stat_card(label, value, color="#B6FF3C") -> str:
    return f'<div class="stat" style="--c:{color}"><div class="k">{label}</div><div class="v">{value}</div></div>'


def neon(fig):
    """Dark, neon styling for a Plotly chart: st.plotly_chart(neon(fig))."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Space Grotesk", color="#E8ECF5"),
                      colorway=["#B6FF3C", "#6AA8FF", "#FFB84D", "#FF6B7A"], margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(gridcolor="#232B3F")
    fig.update_yaxes(gridcolor="#232B3F")
    return fig