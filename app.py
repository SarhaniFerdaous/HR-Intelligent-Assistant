import os
import tempfile
import streamlit as st
from agent import (
    run_agent, build_history,
    save_feedback, load_feedback_stats,
    suggest_questions_for_doc,
)
from ingest import ingest_file, ingest_folder, list_documents, delete_document

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Dimploy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Sidebar toggle */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
}

/* Sidebar — let Streamlit handle collapse/expand naturally */
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
    transition: all 0.3s ease !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Main content responds when sidebar collapses */
.main .block-container {
    max-width: 780px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin: 0 auto !important;
    transition: all 0.3s ease !important;
}

/* Buttons — base */
.stButton > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    padding: 6px 14px !important;
}

/* Delete doc button — small, borderless, subtle */
[data-testid="stSidebar"] .stButton > button {
    padding: 2px 8px !important;
    font-size: 11px !important;
    border: none !important;
    background: transparent !important;
    opacity: 0.35 !important;
    min-width: unset !important;
    width: auto !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 1 !important;
    background: rgba(239,68,68,0.12) !important;
    color: #f87171 !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    border-radius: 14px !important;
}

/* Chat messages */
[data-testid="stChatMessage"] { padding: 0.3rem 0 !important; }

/* Expander */
[data-testid="stExpander"] { border-radius: 10px !important; }

/* Code */
code, pre { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { border-radius: 4px; opacity: 0.5; }

/* ── Sidebar components ── */
.brand-wrap  { margin-bottom: 4px; }
.brand-name  { font-size: 17px; font-weight: 600; letter-spacing: -0.3px; }
.brand-sub   { font-size: 11px; opacity: 0.4; margin-top: 1px; }

.sec-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; margin: 14px 0 7px;
    display: block;
}

/* Doc card */
.doc-card {
    border-radius: 8px; padding: 7px 10px; margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.03);
}
.doc-name  {
    font-size: 12px; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 200px; display: block;
}
.doc-meta  { font-size: 10px; opacity: 0.38; margin-top: 1px; }

/* Stats */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; margin: 6px 0; }
.stat-box  {
    border-radius: 8px; padding: 9px 4px; text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.03);
}
.stat-num { font-size: 20px; font-weight: 600; line-height: 1; }
.stat-lbl { font-size: 10px; opacity: 0.38; margin-top: 2px; }

/* Hamburger panel */
.hb-panel {
    border-radius: 10px; padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.03);
    margin-top: 4px;
}
.hb-title { font-size: 11px; font-weight: 600; opacity: 0.5; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }

.tool-row  { font-size: 12px; opacity: 0.6; padding: 3px 0; display: flex; align-items: center; gap: 8px; }
.ticket-row { font-size: 12px; opacity: 0.6; padding: 2px 0; display: flex; justify-content: space-between; }

.pill       { font-size: 10px; font-weight: 600; padding: 2px 7px; border-radius: 20px; letter-spacing: 0.03em; white-space: nowrap; }
.pill-rag   { background: rgba(34,197,94,0.15);  color: #22c55e; }
.pill-check { background: rgba(245,158,11,0.15); color: #f59e0b; }
.pill-api   { background: rgba(99,102,241,0.15); color: #818cf8; }

/* ── Response components ── */
.meta-row  { display: flex; align-items: center; gap: 6px; margin-top: 10px; flex-wrap: wrap; }
.meta-chip {
    font-size: 11px; padding: 3px 10px; border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.05);
    display: inline-flex; align-items: center; gap: 5px;
}
.conf-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }

/* Feedback buttons row */
.fb-row { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
.fb-btn {
    font-size: 12px; padding: 5px 14px; border-radius: 20px; cursor: pointer;
    border: 1px solid rgba(255,255,255,0.12);
    background: transparent; color: inherit; opacity: 0.65;
    transition: all 0.15s; font-family: 'DM Sans', sans-serif;
}
.fb-btn:hover { opacity: 1; border-color: rgba(255,255,255,0.3); background: rgba(255,255,255,0.06); }

.escalation-bar {
    margin-top: 10px; padding: 8px 12px; border-radius: 8px;
    font-size: 12px; opacity: 0.55;
    border: 1px solid rgba(255,255,255,0.06);
}
.escalation-bar a { text-decoration: none; font-weight: 500; }

/* Welcome */
.welcome-wrap  { max-width: 680px; margin: 52px auto 28px; padding: 0 8px; }
.welcome-title { font-size: 26px; font-weight: 600; letter-spacing: -0.4px; margin-bottom: 6px; }
.welcome-sub   { font-size: 14px; opacity: 0.5; margin-bottom: 28px; line-height: 1.6; }
.sug-label     { font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.35; margin-bottom: 10px; }

.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 12px 0; }

.error-box {
    padding: 11px 15px; border-radius: 10px; margin-top: 6px;
    border: 1px solid rgba(239,68,68,0.25);
    background: rgba(239,68,68,0.07);
    font-size: 13px; color: #f87171;
}

/* Reasoning step */
.step-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.step-num    { font-size: 11px; font-weight: 600; opacity: 0.4; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

_DEFAULTS = {
    "messages":       [],
    "ingested_files": set(),
    "default_loaded": False,
    "suggested_qs":   [],
    "feedback_given": set(),
    "ticket_counts":  {},
    "pending_input":  None,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Sync ingested_files with ChromaDB on fresh load
if not st.session_state.ingested_files:
    st.session_state.ingested_files = {d["name"] for d in list_documents()}

# ─────────────────────────────────────────────
# LOAD DEFAULT DOCS
# ─────────────────────────────────────────────

if not st.session_state.default_loaded:
    if os.path.exists("docs") and os.listdir("docs"):
        existing = {d["name"] for d in list_documents()}
        if not existing:
            with st.spinner("Loading documents…"):
                ingest_folder("docs")
    st.session_state.default_loaded = True

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TICKET_ICONS = {
    "Leave": "🌴", "Payroll": "💰", "Onboarding": "👋",
    "Compliance": "⚖️", "Training": "📚", "Benefits": "❤️",
    "Urgent": "🚨", "General": "💬",
}

# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────

def render_assistant_msg(msg: dict, idx: int, user_question: str = "") -> None:

    if msg.get("error"):
        st.markdown(
            f"<div class='error-box'>⚠️ {msg.get('content') or msg.get('output', 'An error occurred.')}</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(msg["content"])

    conf   = msg.get("conf") or {}
    ticket = msg.get("ticket", "General")
    steps  = msg.get("steps", [])

    # Confidence + ticket
    if conf:
        conf_color = conf.get("color", "#8B90A0")
        conf_level = conf.get("level", "")
        icon       = TICKET_ICONS.get(ticket, "💬")
        st.markdown(f"""
<div class='meta-row'>
    <span class='meta-chip'>
        <span class='conf-dot' style='background:{conf_color}'></span>
        {conf_level}
    </span>
    <span class='meta-chip'>{icon} {ticket}</span>
</div>
""", unsafe_allow_html=True)

    # Reasoning steps — collapsed, clean
    if steps:
        with st.expander("View reasoning steps"):
            for i, (action, observation) in enumerate(steps, 1):
                tool_name = action.tool
                if "search" in tool_name:
                    pill = "<span class='pill pill-rag'>RAG</span>"
                elif "compliance" in tool_name:
                    pill = "<span class='pill pill-check'>CHECK</span>"
                else:
                    pill = "<span class='pill pill-api'>API</span>"
                st.markdown(
                    f"<div class='step-header'>"
                    f"<span class='step-num'>Step {i}</span> {pill} "
                    f"<code style='font-size:11px'>{tool_name}</code></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"↳ {action.tool_input}")
                st.code(str(observation)[:500], language="text")
                if i < len(steps):
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Escalation
    st.markdown(
        "<div class='escalation-bar'>"
        "Need a human? 📧 <a href='mailto:hr@company.com'>hr@company.com</a>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Feedback
    msg_key = f"fb_{idx}"
    if msg_key not in st.session_state.feedback_given:
        fb1, fb2, _ = st.columns([1, 1, 5])
        with fb1:
            if st.button("Helpful", key=f"up_{idx}"):
                save_feedback(user_question, msg["content"], "helpful")
                st.session_state.feedback_given.add(msg_key)
                st.rerun()
        with fb2:
            if st.button("Not helpful", key=f"dn_{idx}"):
                save_feedback(user_question, msg["content"], "not_helpful")
                st.session_state.feedback_given.add(msg_key)
                st.rerun()
    else:
        st.caption("✓ Thanks for the feedback")


def _store_and_render(result: dict, user_input: str) -> None:
    from agent import get_confidence, classify_ticket
    steps  = result.get("steps", [])
    conf   = result.get("conf")   or get_confidence(steps)
    ticket = result.get("ticket") or classify_ticket(user_input, result.get("output", ""))

    st.session_state.ticket_counts[ticket] = (
        st.session_state.ticket_counts.get(ticket, 0) + 1
    )
    entry = {
        "role":    "assistant",
        "content": result.get("output", ""),
        "steps":   steps,
        "conf":    conf,
        "ticket":  ticket,
        "error":   result.get("error"),
    }
    st.session_state.messages.append(entry)
    render_assistant_msg(entry, idx=len(st.session_state.messages) - 1, user_question=user_input)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:

    # Brand
    st.markdown(
        "<div class='brand-wrap'>"
        "<div class='brand-name'>🧠 Dimploy</div>"
        "<div class='brand-sub'>HR Intelligence Assistant</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    
    # ── Documents ──
    st.markdown("<span class='sec-label'>Documents</span>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Upload",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Processing {uf.name}…"):
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    real_path = os.path.join(tempfile.gettempdir(), uf.name)
                    os.replace(tmp_path, real_path)
                    result = ingest_file(real_path)
                if result["success"]:
                    st.session_state.ingested_files.add(uf.name)
                    st.session_state.cached_docs = list_documents()
                    st.success(f"✓ {uf.name} — {result['chunks']} chunks")
                    # Suggestions skipped — uses default questions instead
                else:
                    st.error(f"✗ {uf.name}: {result['message']}")

    # Doc list
    # Use cached doc list — only hits ChromaDB when something changes
    if "cached_docs" not in st.session_state:
        st.session_state.cached_docs = list_documents()
    docs = st.session_state.cached_docs

    if docs:
        for doc in docs:
            col_doc, col_del = st.columns([5, 1])
            with col_doc:
                name = doc['name']
                short = name if len(name) <= 22 else name[:20] + "…"
                st.markdown(
                    f"<div class='doc-card'>"
                    f"<div class='doc-name' title='{name}'>📄 {short}</div>"
                    f"<div class='doc-meta'>{doc['chunks']} chunks</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("✕", key=f"del_{doc['name']}"):
                    res = delete_document(doc["name"])
                    if res["success"]:
                        st.session_state.ingested_files.discard(doc["name"])
                        st.session_state.cached_docs = list_documents()
                        st.rerun()
    else:
        st.caption("No documents loaded yet.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.suggested_qs  = []
        st.session_state.ticket_counts = {}
        st.rerun()
# Session stats
    stats   = load_feedback_stats()
    q_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown("<span class='sec-label'>Session</span>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='stat-grid'>
    <div class='stat-box'>
        <div class='stat-num'>{q_count}</div>
        <div class='stat-lbl'>Asked</div>
    </div>
    <div class='stat-box'>
        <div class='stat-num' style='color:#22c55e'>{stats['helpful']}</div>
        <div class='stat-lbl'>Helpful</div>
    </div>
    <div class='stat-box'>
        <div class='stat-num' style='color:#f59e0b'>{stats['not_helpful']}</div>
        <div class='stat-lbl'>Not helpful</div>
    </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.ticket_counts:
        for cat, cnt in sorted(
            st.session_state.ticket_counts.items(), key=lambda x: x[1], reverse=True
        ):
            icon = TICKET_ICONS.get(cat, "💬")
            st.markdown(
                f"<div class='ticket-row'><span>{icon} {cat}</span><strong>{cnt}</strong></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Agent tools
    st.markdown("<span class='sec-label'>Agent Tools</span>", unsafe_allow_html=True)
    st.markdown("""
<div class='tool-row'><span class='pill pill-rag'>RAG</span> Document search</div>
<div class='tool-row'><span class='pill pill-check'>CHECK</span> Compliance</div>
<div class='tool-row'><span class='pill pill-api'>API</span> HR contacts</div>
<div class='tool-row'><span class='pill pill-api'>API</span> Public holidays</div>
<div class='tool-row'><span class='pill pill-api'>API</span> Currency</div>
""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PENDING INPUT
# ─────────────────────────────────────────────

if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None
    st.session_state.messages.append({
        "role": "user", "content": user_input,
        "steps": [], "conf": None, "ticket": None, "error": None,
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history = build_history(st.session_state.messages[:-1])
            result  = run_agent(user_input, history)
        _store_and_render(result, user_input)
    st.rerun()

# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────

if not st.session_state.messages:
    suggestions = st.session_state.suggested_qs or [
        "What do I need to bring on my first day?",
        "How many annual leave days do I get?",
        "I have been sick for 4 days, is that a problem?",
        "Convert 3000 EUR to TND",
        "Public holidays in Tunisia this year",
        "Who handles payroll queries?",
    ]
    st.markdown("""
<div class='welcome-wrap'>
    <div class='welcome-title'>Hello, I'm Dimploy</div>
    <div class='welcome-sub'>
        Your HR intelligence assistant — ask me anything about company policies,
        onboarding, benefits, compliance, or payroll.
    </div>
    <div class='sug-label'>Try asking</div>
</div>
""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, sug in enumerate(suggestions[:6]):
        with (col1 if i % 2 == 0 else col2):
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_input = sug
                st.rerun()

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            prev_q = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""
            render_assistant_msg(msg, idx, user_question=prev_q)

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────

if user_input := st.chat_input("Ask Dimploy anything about HR…"):
    st.session_state.messages.append({
        "role": "user", "content": user_input,
        "steps": [], "conf": None, "ticket": None, "error": None,
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            history = build_history(st.session_state.messages[:-1])
            result  = run_agent(user_input, history)
        _store_and_render(result, user_input)