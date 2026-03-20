"""
main.py
-------
Streamlit UI for the Cognitive Guardrail AI system.

Features:
    - Passcode-based login (persistent user identity across sessions)
    - Continuous chat loop (multiple turns per session)
    - End Session button to close and save session
    - Compare mode: side-by-side with vs without memory
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import sys
import uuid
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import streamlit.components.v1

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Cognitive Guardrail AI",
    page_icon  = "🧠",
    layout     = "wide",
)

# ── Imports ───────────────────────────────────────────────────────────────────
from training.csv_loader import load_merged
from agents.analyst import run_analyst
from agents.devils_advocate import run_consensus_loop
from agents.responder import run_responder
from agents.memory_architect import process_session_end
from memory.fact_vault import search_facts
from memory.inference_layer import search_inferences
from meta.reviewer import (
    log_session_result,
    get_analyst_patch,
    get_responder_patch,
    get_session_count,
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg         : #0f0f13;
    --bg-card    : #16161d;
    --bg-panel   : #1c1c26;
    --accent     : #c8b8ff;
    --accent-2   : #82e8c0;
    --accent-warm: #ffb86c;
    --danger     : #ff6b6b;
    --text       : #e8e4f0;
    --text-muted : #7a7490;
    --border     : #2a2840;
    --radius     : 14px;
}

html, body, [class*="css"] {
    font-family : 'DM Sans', sans-serif;
    background  : var(--bg);
    color       : var(--text);
}

.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }

.cg-header {
    display        : flex;
    align-items    : baseline;
    gap            : 16px;
    margin-bottom  : 2rem;
    border-bottom  : 1px solid var(--border);
    padding-bottom : 1.5rem;
}
.cg-title {
    font-family   : 'DM Serif Display', serif;
    font-size     : 2.2rem;
    color         : var(--text);
    margin        : 0;
    letter-spacing: -0.5px;
}
.cg-title span { color: var(--accent); font-style: italic; }
.cg-subtitle {
    font-size  : 0.85rem;
    color      : var(--text-muted);
    font-family: 'DM Mono', monospace;
    margin     : 0;
}

/* Login box */
.login-wrap {
    max-width     : 420px;
    margin        : 6rem auto;
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-radius : var(--radius);
    padding       : 2.5rem;
    text-align    : center;
}
.login-title {
    font-family   : 'DM Serif Display', serif;
    font-size     : 1.8rem;
    color         : var(--accent);
    margin-bottom : 0.4rem;
}
.login-sub {
    font-size     : 0.85rem;
    color         : var(--text-muted);
    margin-bottom : 1.8rem;
    font-family   : 'DM Mono', monospace;
}

/* Cards */
.cg-card {
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-radius : var(--radius);
    padding       : 1.4rem 1.6rem;
    margin-bottom : 1rem;
}
.cg-card-accent { border-left: 3px solid var(--accent); }
.cg-card-green  { border-left: 3px solid var(--accent-2); }
.cg-card-warm   { border-left: 3px solid var(--accent-warm); }
.cg-card-danger { border-left: 3px solid var(--danger); background: #1f1520; }

.cg-label {
    font-family   : 'DM Mono', monospace;
    font-size     : 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color         : var(--text-muted);
    margin-bottom : 0.4rem;
}
.cg-value { font-size: 1rem; color: var(--text); line-height: 1.6; }

.distortion-badge {
    display       : inline-block;
    background    : #1e1830;
    border        : 1px solid var(--accent);
    color         : var(--accent);
    font-family   : 'DM Mono', monospace;
    font-size     : 0.75rem;
    padding       : 4px 12px;
    border-radius : 20px;
    margin-bottom : 0.5rem;
}
.distortion-badge-none {
    border-color: var(--accent-2);
    color       : var(--accent-2);
    background  : #101e18;
}

.conf-bar-wrap {
    background    : var(--bg-panel);
    border-radius : 4px;
    height        : 6px;
    width         : 100%;
    margin-top    : 6px;
}
.conf-bar-fill {
    height        : 6px;
    border-radius : 4px;
    background    : linear-gradient(90deg, var(--accent), var(--accent-2));
}

/* Chat bubbles */
.chat-bubble-user {
    background    : var(--bg-panel);
    border-radius : 12px 12px 4px 12px;
    padding       : 0.8rem 1.2rem;
    margin-bottom : 0.5rem;
    font-size     : 0.95rem;
    max-width     : 75%;
    margin-left   : auto;
    text-align    : right;
    border        : 1px solid var(--border);
}
.chat-bubble-ai {
    background    : #1a1830;
    border-radius : 12px 12px 12px 4px;
    padding       : 0.8rem 1.2rem;
    margin-bottom : 0.5rem;
    font-size     : 0.95rem;
    max-width     : 75%;
    border        : 1px solid var(--border);
    border-left   : 2px solid var(--accent);
}
.chat-meta {
    font-family   : 'DM Mono', monospace;
    font-size     : 0.65rem;
    color         : var(--text-muted);
    margin-bottom : 0.3rem;
}

/* Fact pills */
.fact-pill {
    display       : inline-block;
    background    : #101e18;
    border        : 1px solid var(--accent-2);
    color         : var(--accent-2);
    font-size     : 0.78rem;
    padding       : 3px 10px;
    border-radius : 12px;
    margin        : 3px;
    font-family   : 'DM Mono', monospace;
}

/* Compare columns */
.compare-col-label {
    font-family    : 'DM Mono', monospace;
    font-size      : 0.7rem;
    text-transform : uppercase;
    letter-spacing : 2px;
    margin-bottom  : 1rem;
    padding-bottom : 0.6rem;
    border-bottom  : 1px solid var(--border);
}
.with-memory    { color: var(--accent); }
.without-memory { color: var(--accent-warm); }

/* Session ended banner */
.session-ended {
    background    : #101e18;
    border        : 1px solid var(--accent-2);
    border-radius : var(--radius);
    padding       : 1.5rem;
    text-align    : center;
    margin-top    : 2rem;
}

/* Streamlit overrides */
.stTextArea textarea {
    background    : var(--bg-panel) !important;
    border        : 1px solid var(--border) !important;
    color         : var(--text) !important;
    border-radius : 10px !important;
    font-family   : 'DM Sans', sans-serif !important;
    font-size     : 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color : var(--accent) !important;
    box-shadow   : 0 0 0 2px rgba(200,184,255,0.15) !important;
}
.stButton > button {
    background    : var(--accent) !important;
    color         : #0f0f13 !important;
    border        : none !important;
    border-radius : 8px !important;
    font-family   : 'DM Sans', sans-serif !important;
    font-weight   : 600 !important;
    padding       : 0.5rem 1.4rem !important;
    transition    : opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stSidebarContent"] {
    background  : var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Dataset (cached) ──────────────────────────────────────────────────────────
@st.cache_resource
def get_dataframe():
    return load_merged()


# ── Auth import ──────────────────────────────────────────────────────────────
from auth import register_user, login_user
from stats_tracker import get_user_stats, get_global_stats


# ── Session state init ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "logged_in"      : False,
        "user_id"        : None,
        "username"       : None,
        "auth_mode"      : "login",   # "login" or "register"
        "current_view"   : "chat",    # "chat" or "stats"
        "session_id"     : str(uuid.uuid4())[:8],
        "chat_history"   : [],      # list of {user, response, label, technique}
        "confirmed_facts": [],
        "last_result"    : None,
        "last_result_nm" : None,
        "compare_mode"   : False,
        "session_ended"  : False,
        "turn_count"     : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Pipeline runners ──────────────────────────────────────────────────────────

def run_full_pipeline(user_input: str, user_id: str, session_id: str) -> dict:
    df = get_dataframe()

    memory_facts      = search_facts(user_id, user_input, n_results=5)
    recent_inferences = search_inferences(user_id, user_input, n_results=3)
    analyst_patch     = get_analyst_patch(user_id)

    analyst_result = run_analyst(
        user_input      = user_input,
        df              = df,
        k               = 5,
        override_prompt = analyst_patch or None,
    )

    final_analyst, final_devils, loops = run_consensus_loop(
        user_input     = user_input,
        analyst_result = analyst_result,
        df             = df,
        memory_facts   = memory_facts,
        max_loops      = 2,
    )

    responder_result = run_responder(
        user_input        = user_input,
        analyst_result    = final_analyst,
        df                = df,
        recent_inferences = recent_inferences,
        k                 = 3,
    )

    conversation = [
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": responder_result["response"]},
    ]

    architect_result = process_session_end(
        user_id          = user_id,
        session_id       = session_id,
        conversation     = conversation,
        analyst_result   = final_analyst,
        responder_result = responder_result,
        user_feedback    = None,
    )

    log_session_result(
        user_id         = user_id,
        session_id      = session_id,
        analyst_label   = final_analyst.get("label", ""),
        da_verdict      = final_devils.get("verdict", ""),
        da_suggested    = final_devils.get("suggested_label", ""),
        consensus_loops = loops,
        user_feedback   = None,
        flagged         = responder_result.get("flagged", False),
        technique       = responder_result.get("technique", ""),
    )

    return {
        "response"       : responder_result["response"],
        "follow_up"      : responder_result["follow_up"],
        "label"          : final_analyst["label"],
        "confidence"     : final_analyst["confidence"],
        "distorted_part" : final_analyst.get("distorted_part", ""),
        "explanation"    : final_analyst.get("explanation", ""),
        "reality_check"  : final_analyst.get("reality_check", ""),
        "technique"      : responder_result["technique"],
        "flagged"        : responder_result["flagged"],
        "flag_reason"    : responder_result.get("flag_reason", ""),
        "da_verdict"     : final_devils["verdict"],
        "consensus_loops": loops,
        "memory_facts"   : memory_facts,
        "ai_distortions" : architect_result.get("ai_distortions_found", []),
    }


def run_pipeline_no_memory(user_input: str) -> dict:
    df = get_dataframe()

    analyst_result = run_analyst(user_input=user_input, df=df, k=5)

    final_analyst, final_devils, loops = run_consensus_loop(
        user_input     = user_input,
        analyst_result = analyst_result,
        df             = df,
        memory_facts   = [],
        max_loops      = 2,
    )

    responder_result = run_responder(
        user_input        = user_input,
        analyst_result    = final_analyst,
        df                = df,
        recent_inferences = [],
        k                 = 3,
    )

    return {
        "response"       : responder_result["response"],
        "follow_up"      : responder_result["follow_up"],
        "label"          : final_analyst["label"],
        "confidence"     : final_analyst["confidence"],
        "distorted_part" : final_analyst.get("distorted_part", ""),
        "explanation"    : final_analyst.get("explanation", ""),
        "technique"      : responder_result["technique"],
        "flagged"        : responder_result["flagged"],
        "da_verdict"     : final_devils["verdict"],
        "consensus_loops": loops,
    }


# ── UI Components ─────────────────────────────────────────────────────────────

def render_result_card(result: dict, with_memory: bool = True):
    label    = result.get("label", "")
    conf     = result.get("confidence", 0.0)
    is_none  = label == "No distortion"
    badge_cls = "distortion-badge-none" if is_none else "distortion-badge"

    st.markdown(
        f'<div class="{badge_cls}">{"✓ " if is_none else "⚠ "}{label}</div>',
        unsafe_allow_html=True,
    )

    conf_pct = int(conf * 100)
    st.markdown(
        f'<div class="cg-label">Confidence — {conf_pct}%</div>'
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill" style="width:{conf_pct}%"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if result.get("distorted_part"):
        st.markdown(
            f'<div class="cg-label">Distorted phrase</div>'
            f'<div class="cg-value" style="color:#ffb86c;font-style:italic;">'
            f'"{result["distorted_part"]}"</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    if result.get("explanation"):
        st.markdown(
            f'<div class="cg-label">Why this is a distortion</div>'
            f'<div class="cg-value">{result["explanation"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="cg-label">Response</div>'
        f'<div class="cg-value" style="line-height:1.8;">{result["response"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if result.get("follow_up"):
        st.markdown(
            f'<div class="cg-label">Follow-up question</div>'
            f'<div class="cg-value" style="color:var(--accent);font-style:italic;">'
            f'{result["follow_up"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="cg-label">CBT Technique</div>'
            f'<div class="cg-value" style="font-size:0.9rem;">{result.get("technique","—")}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        verdict       = result.get("da_verdict", "")
        verdict_color = "#82e8c0" if verdict == "AGREE" else "#ffb86c"
        loops         = result.get("consensus_loops", 0)
        st.markdown(
            f'<div class="cg-label">Devil\'s Advocate</div>'
            f'<div class="cg-value" style="font-size:0.9rem;color:{verdict_color};">'
            f'{verdict} ({loops} loop{"s" if loops != 1 else ""})</div>',
            unsafe_allow_html=True,
        )

    if with_memory and result.get("memory_facts"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="cg-label">Memory facts used</div>', unsafe_allow_html=True)
        pills = "".join([
            f'<span class="fact-pill">{f[:60]}{"..." if len(f)>60 else ""}</span>'
            for f in result["memory_facts"]
        ])
        st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)

    if result.get("flagged"):
        st.markdown(
            f'<div class="cg-card cg-card-danger" style="margin-top:1rem;">'
            f'<div class="cg-label" style="color:var(--danger);">⚠ Crisis flag</div>'
            f'<div class="cg-value">{result.get("flag_reason","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_chat_history():
    """Renders all previous turns in the session."""
    if not st.session_state["chat_history"]:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="cg-label" style="margin-bottom:0.8rem;">'
        f'Conversation — {len(st.session_state["chat_history"])} turn(s)</div>',
        unsafe_allow_html=True,
    )

    for turn in st.session_state["chat_history"]:
        # User bubble
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;">'
            f'<div class="chat-bubble-user">{turn["user"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # AI bubble
        badge = turn.get("label", "")
        tech  = turn.get("technique", "")
        st.markdown(
            f'<div class="chat-bubble-ai">'
            f'<div class="chat-meta">{badge} · {tech}</div>'
            f'{turn["response"]}'
            f'<div style="color:var(--accent);font-style:italic;font-size:0.88rem;margin-top:0.5rem;">'
            f'{turn.get("follow_up","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _build_donut_chart(labels: list, values: list, palette: list, chart_id: str) -> str:
    """Builds a Chart.js donut chart as an HTML string."""
    import json as _json
    short_labels = [l[:22] + "..." if len(l) > 22 else l for l in labels]
    return f"""
<html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head><body style="background:transparent;margin:0;padding:0;">
<canvas id="{chart_id}" style="max-height:300px;"></canvas>
<script>
new Chart(document.getElementById('{chart_id}'), {{
  type: 'doughnut',
  data: {{
    labels: {_json.dumps(short_labels)},
    datasets: [{{
      data: {_json.dumps(values)},
      backgroundColor: {_json.dumps(palette[:len(values)])},
      borderWidth: 2,
      borderColor: '#0f0f13',
      hoverOffset: 8,
    }}]
  }},
  options: {{
    responsive: true,
    cutout: '62%',
    plugins: {{
      legend: {{
        position: 'right',
        labels: {{
          color: '#e8e4f0',
          font: {{ size: 11, family: 'DM Sans' }},
          boxWidth: 12,
          padding: 10,
        }}
      }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}} (${{Math.round(ctx.parsed/ctx.dataset.data.reduce((a,b)=>a+b,0)*100)}}%)`
        }}
      }}
    }}
  }}
}});
</script></body></html>"""


def _build_bar_chart(labels: list, values: list, palette: list, chart_id: str) -> str:
    """Builds a Chart.js horizontal bar chart as an HTML string."""
    import json as _json
    short_labels = [l[:20] + "..." if len(l) > 20 else l for l in labels]
    return f"""
<html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head><body style="background:transparent;margin:0;padding:0;">
<canvas id="{chart_id}" style="max-height:300px;"></canvas>
<script>
new Chart(document.getElementById('{chart_id}'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(short_labels)},
    datasets: [{{
      label: 'Count',
      data: {_json.dumps(values)},
      backgroundColor: {_json.dumps(palette[:len(values)])},
      borderRadius: 6,
      borderWidth: 0,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.parsed.x}} sessions`
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#7a7490', font: {{ size: 11 }} }},
        grid:  {{ color: '#2a2840' }},
      }},
      y: {{
        ticks: {{ color: '#e8e4f0', font: {{ size: 11, family: 'DM Sans' }} }},
        grid:  {{ display: false }},
      }}
    }}
  }}
}});
</script></body></html>"""


def _build_line_chart(dates: list, counts: list, chart_id: str) -> str:
    """Builds a Chart.js line chart for session activity over time."""
    import json as _json
    return f"""
<html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
</head><body style="background:transparent;margin:0;padding:0;">
<canvas id="{chart_id}" style="max-height:200px;"></canvas>
<script>
new Chart(document.getElementById('{chart_id}'), {{
  type: 'line',
  data: {{
    labels: {_json.dumps(dates)},
    datasets: [{{
      label: 'Sessions',
      data: {_json.dumps(counts)},
      borderColor: '#c8b8ff',
      backgroundColor: 'rgba(200,184,255,0.12)',
      borderWidth: 2,
      pointBackgroundColor: '#c8b8ff',
      pointRadius: 4,
      fill: true,
      tension: 0.4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.parsed.y}} session${{ctx.parsed.y !== 1 ? 's' : ''}}`
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#7a7490', font: {{ size: 10 }}, maxRotation: 30 }},
        grid:  {{ color: '#2a2840' }},
      }},
      y: {{
        ticks: {{ color: '#7a7490', font: {{ size: 10 }}, stepSize: 1 }},
        grid:  {{ color: '#2a2840' }},
        beginAtZero: true,
      }}
    }}
  }}
}});
</script></body></html>"""


def _render_bar(label: str, count: int, total: int, color: str = "var(--accent)"):
    """Renders a horizontal bar for a distortion label."""
    pct = round(count / total * 100) if total > 0 else 0
    st.markdown(
        f'<div style="margin-bottom:0.7rem;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:0.82rem;color:var(--text);">{label}</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:var(--text-muted);">'
        f'{count} · {pct}%</span></div>'
        f'<div style="background:var(--bg-panel);border-radius:4px;height:8px;">'
        f'<div style="width:{pct}%;height:8px;border-radius:4px;background:{color};'
        f'transition:width 0.4s ease;"></div></div></div>',
        unsafe_allow_html=True,
    )


def render_user_stats():
    """Renders the per-user stats dashboard — only shown when user navigates here."""
    uid   = st.session_state["user_id"]
    uname = st.session_state.get("username", uid)
    stats = get_user_stats(uid)

    st.markdown(
        f'<div class="cg-header">'
        f'<h1 class="cg-title">Your <span>Stats</span></h1>'
        f'<p class="cg-subtitle">{uname} · personal analytics</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if stats.get("total_sessions", 0) == 0:
        st.markdown(
            '<div class="cg-card" style="text-align:center;padding:3rem;">'
            '<div style="font-size:1.1rem;color:var(--text-muted);">No sessions yet.</div>'
            '<div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.5rem;">'
            'Complete a chat session to see your personal stats here.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Top metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Sessions",       stats["total_sessions"],       "var(--accent)"),
        (c2, "Distortions Detected", stats["total_distortions"],    "var(--accent-warm)"),
        (c3, "Helpful Rate",         f'{stats["helpful_rate"]}%',   "var(--accent-2)"),
        (c4, "Clean Sessions",       stats["no_distortion_count"],  "var(--accent-2)"),
    ]:
        with col:
            st.markdown(
                f'<div class="cg-card" style="text-align:center;">'
                f'<div class="cg-label">{label}</div>'
                f'<div style="font-size:2rem;font-weight:700;color:{color};">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    if stats["distortion_counts"]:
        chart_left, chart_right = st.columns(2)

        # Donut chart — distortion breakdown
        with chart_left:
            st.markdown(
                '<div class="cg-card cg-card-accent">'
                '<div class="cg-label" style="margin-bottom:0.5rem;">Distortion Breakdown</div>',
                unsafe_allow_html=True,
            )
            labels  = list(stats["distortion_counts"].keys())
            values  = list(stats["distortion_counts"].values())
            palette = ["#c8b8ff","#ffb86c","#82e8c0","#ff9eb5","#7ec8e3",
                       "#f5c842","#b388ff","#80cbc4","#ef9a9a","#ce93d8"]

            donut_html = _build_donut_chart(labels, values, palette, "user_donut")
            st.components.v1.html(donut_html, height=320)
            st.markdown('</div>', unsafe_allow_html=True)

        # Bar chart — same data horizontal
        with chart_right:
            st.markdown(
                '<div class="cg-card">'
                '<div class="cg-label" style="margin-bottom:0.5rem;">Count per Distortion Type</div>',
                unsafe_allow_html=True,
            )
            bar_html = _build_bar_chart(labels, values, palette, "user_bar")
            st.components.v1.html(bar_html, height=320)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Timeline chart ────────────────────────────────────────────────────────
    if stats.get("timeline") and len(stats["timeline"]) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="cg-card"><div class="cg-label" style="margin-bottom:0.5rem;">'
            'Session Activity Over Time</div>',
            unsafe_allow_html=True,
        )

        # Count sessions per date
        from collections import Counter as _Counter
        date_counts = _Counter(t["timestamp"] for t in stats["timeline"] if t.get("timestamp"))
        dates  = sorted(date_counts.keys())
        counts = [date_counts[d] for d in dates]

        line_html = _build_line_chart(dates, counts, "user_timeline")
        st.components.v1.html(line_html, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right panel ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Text breakdown bars
        st.markdown(
            '<div class="cg-card cg-card-accent">'
            '<div class="cg-label" style="margin-bottom:1rem;">Breakdown with Percentages</div>',
            unsafe_allow_html=True,
        )
        if stats["distortion_counts"]:
            total_dist = stats["total_distortions"]
            colors = ["var(--accent)","var(--accent-warm)","var(--accent-2)",
                      "#ff9eb5","#7ec8e3","#f5c842","#b388ff","#80cbc4","#ef9a9a","#ce93d8"]
            for i, (lbl, cnt) in enumerate(stats["distortion_counts"].items()):
                _render_bar(lbl, cnt, total_dist, colors[i % len(colors)])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        top = stats.get("top_distortion", "None")
        pct = stats.get("distortion_pct", {}).get(top, 0)
        st.markdown(
            f'<div class="cg-card cg-card-warm" style="margin-bottom:1rem;">'
            f'<div class="cg-label">Most Common Pattern</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:var(--accent-warm);">{top}</div>'
            f'<div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">'
            f'Detected in {pct}% of distorted sessions</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cg-card" style="margin-bottom:1rem;">'
            f'<div class="cg-label">Devil\'s Advocate Override Rate</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:var(--accent);">'
            f'{stats["da_challenge_rate"]}%</div>'
            f'<div style="font-size:0.8rem;color:var(--text-muted);margin-top:0.3rem;">'
            f'sessions where initial label was challenged</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if stats.get("techniques_used"):
            st.markdown(
                '<div class="cg-card"><div class="cg-label" style="margin-bottom:0.8rem;">'
                'CBT Techniques Used</div>',
                unsafe_allow_html=True,
            )
            for tech, count in list(stats["techniques_used"].items())[:5]:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:4px 0;border-bottom:1px solid var(--border);">'
                    f'<span style="font-size:0.82rem;">{tech}</span>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.75rem;'
                    f'color:var(--accent);">{count}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Session log table ─────────────────────────────────────────────────────
    if stats.get("timeline"):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Session log"):
            for turn in reversed(stats["timeline"]):
                lbl      = turn.get("label", "")
                date     = turn.get("timestamp", "")
                feedback = turn.get("feedback", "none")
                verdict  = turn.get("verdict", "")
                lbl_color  = "var(--accent-2)" if lbl == "No distortion" else "var(--accent-warm)"
                fb_color   = "var(--accent-2)" if feedback == "helpful" else "var(--text-muted)"
                da_color   = "var(--danger)" if verdict == "CHALLENGE" else "var(--accent-2)"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:6px 0;border-bottom:1px solid var(--border);">'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                    f'color:var(--text-muted);width:90px;">{date}</span>'
                    f'<span style="font-size:0.82rem;color:{lbl_color};flex:1;padding:0 1rem;">{lbl}</span>'
                    f'<span style="font-size:0.72rem;color:{da_color};width:80px;">{verdict}</span>'
                    f'<span style="font-size:0.72rem;color:{fb_color};width:80px;">{feedback}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_global_stats():
    """Renders the global stats dashboard — only shown when user navigates here."""
    stats = get_global_stats()

    st.markdown(
        '<div class="cg-header">'
        '<h1 class="cg-title">Global <span>Stats</span></h1>'
        '<p class="cg-subtitle">Aggregated across all users · anonymised</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if stats.get("total_users", 0) == 0:
        st.markdown(
            '<div class="cg-card" style="text-align:center;padding:3rem;">'
            '<div style="font-size:1.1rem;color:var(--text-muted);">No data collected yet.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Users",          stats["total_users"],          "var(--accent)"),
        (c2, "Total Sessions",       stats["total_sessions"],       "var(--accent-warm)"),
        (c3, "Distortions Detected", stats["total_distortions"],    "var(--accent-warm)"),
        (c4, "Global Helpful Rate",  f'{stats["helpful_rate"]}%',   "var(--accent-2)"),
    ]:
        with col:
            st.markdown(
                f'<div class="cg-card" style="text-align:center;">'
                f'<div class="cg-label">{label}</div>'
                f'<div style="font-size:2rem;font-weight:700;color:{color};">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    if stats["distortion_counts"]:
        chart_left, chart_right = st.columns(2)

        palette = ["#c8b8ff","#ffb86c","#82e8c0","#ff9eb5","#7ec8e3",
                   "#f5c842","#b388ff","#80cbc4","#ef9a9a","#ce93d8"]
        labels  = list(stats["distortion_counts"].keys())
        values  = list(stats["distortion_counts"].values())

        with chart_left:
            st.markdown(
                '<div class="cg-card cg-card-accent">'
                '<div class="cg-label" style="margin-bottom:0.5rem;">Global Distortion Breakdown</div>',
                unsafe_allow_html=True,
            )
            donut_html = _build_donut_chart(labels, values, palette, "global_donut")
            st.components.v1.html(donut_html, height=320)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_right:
            st.markdown(
                '<div class="cg-card">'
                '<div class="cg-label" style="margin-bottom:0.5rem;">Count per Distortion Type</div>',
                unsafe_allow_html=True,
            )
            bar_html = _build_bar_chart(labels, values, palette, "global_bar")
            st.components.v1.html(bar_html, height=320)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Sessions over time ────────────────────────────────────────────────────
    if stats.get("timeline") and len(stats["timeline"]) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="cg-card"><div class="cg-label" style="margin-bottom:0.5rem;">'
            'Sessions Over Time (All Users)</div>',
            unsafe_allow_html=True,
        )
        dates  = [t["date"] for t in stats["timeline"]]
        counts = [t["count"] for t in stats["timeline"]]
        line_html = _build_line_chart(dates, counts, "global_timeline")
        st.components.v1.html(line_html, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Bottom panels ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            '<div class="cg-card cg-card-accent">'
            '<div class="cg-label" style="margin-bottom:1rem;">Breakdown with Percentages</div>',
            unsafe_allow_html=True,
        )
        if stats["distortion_counts"]:
            total_dist = stats["total_distortions"]
            colors = ["var(--accent)","var(--accent-warm)","var(--accent-2)",
                      "#ff9eb5","#7ec8e3","#f5c842","#b388ff","#80cbc4","#ef9a9a","#ce93d8"]
            for i, (lbl, cnt) in enumerate(stats["distortion_counts"].items()):
                _render_bar(lbl, cnt, total_dist, colors[i % len(colors)])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        top = stats.get("top_distortion", "None")
        pct = stats.get("distortion_pct", {}).get(top, 0)
        st.markdown(
            f'<div class="cg-card cg-card-warm" style="margin-bottom:1rem;">'
            f'<div class="cg-label">Most Common Pattern (Global)</div>'
            f'<div style="font-size:1.1rem;font-weight:600;color:var(--accent-warm);">{top}</div>'
            f'<div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">'
            f'{pct}% of all distorted sessions</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cg-card" style="margin-bottom:1rem;">'
            f'<div class="cg-label">Global DA Override Rate</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:var(--accent);">'
            f'{stats["da_challenge_rate"]}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if stats.get("most_active_users"):
            st.markdown(
                '<div class="cg-card"><div class="cg-label" style="margin-bottom:0.8rem;">'
                'Most Active Users (anonymised)</div>',
                unsafe_allow_html=True,
            )
            for i, u in enumerate(stats["most_active_users"], 1):
                uid_display = u["user_id"][:8] + "..."
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:4px 0;border-bottom:1px solid var(--border);">'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;">'
                    f'#{i} {uid_display}</span>'
                    f'<span style="font-size:0.78rem;color:var(--accent);">'
                    f'{u["sessions"]} sessions</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)




def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'DM Serif Display\',serif;font-size:1.2rem;'
            'color:var(--accent);margin-bottom:1rem;">Session Info</div>',
            unsafe_allow_html=True,
        )

        uid   = st.session_state["user_id"]
        sid   = st.session_state["session_id"]
        cnt   = get_session_count(uid)
        turns = st.session_state["turn_count"]
        label = st.session_state.get("username", uid)

        st.markdown(
            f'<div class="cg-label">User</div>'
            f'<div style="font-size:1rem;font-weight:600;color:var(--text);margin-bottom:1rem;">{label}</div>'
            f'<div class="cg-label">Session ID</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.8rem;'
            f'color:var(--text-muted);margin-bottom:1rem;">{sid}</div>'
            f'<div class="cg-label">Total sessions logged</div>'
            f'<div style="font-size:1.4rem;font-weight:600;color:var(--accent);'
            f'margin-bottom:0.5rem;">{cnt}</div>'
            f'<div class="cg-label">Turns this session</div>'
            f'<div style="font-size:1.4rem;font-weight:600;color:var(--accent-2);'
            f'margin-bottom:1.5rem;">{turns}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Navigation ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="cg-label" style="margin-bottom:0.5rem;">Navigate</div>',
            unsafe_allow_html=True,
        )

        if st.button("💬 Chat", key="nav_chat", use_container_width=True):
            st.session_state["current_view"] = "chat"
            st.rerun()

        if st.button("📊 My Stats", key="nav_my_stats", use_container_width=True):
            st.session_state["current_view"] = "user_stats"
            st.rerun()

        if st.button("🌐 Global Stats", key="nav_global_stats", use_container_width=True):
            st.session_state["current_view"] = "global_stats"
            st.rerun()

        st.divider()

        # End session button
        if not st.session_state["session_ended"]:
            if st.button("⏹ End Session", key="end_session_btn", use_container_width=True):
                st.session_state["session_ended"] = True
                st.rerun()

        # Logout button
        if st.button("↩ Logout", key="logout_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Login screen ──────────────────────────────────────────────────────────────

def render_login():
    st.markdown(
        '<div class="login-wrap">'
        '<div class="login-title">Cognitive <span style="font-style:italic;">Guardrail</span></div>'
        '<div class="login-sub">Sign in or create an account to begin</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col = st.columns([1, 2, 1])[1]
    with col:

        # Toggle between Login and Register
        auth_mode = st.radio(
            "Mode",
            options          = ["Login", "Register"],
            horizontal       = True,
            label_visibility = "collapsed",
            key              = "auth_mode_toggle",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        username = st.text_input(
            "Username",
            placeholder      = "Enter username...",
            label_visibility = "visible",
            key              = "username_input",
        )

        password = st.text_input(
            "Password",
            type             = "password",
            placeholder      = "Enter password...",
            label_visibility = "visible",
            key              = "password_input",
        )

        if auth_mode == "Register":
            confirm_password = st.text_input(
                "Confirm Password",
                type             = "password",
                placeholder      = "Re-enter password...",
                label_visibility = "visible",
                key              = "confirm_password_input",
            )
            st.caption("Password must be 8+ characters with at least one letter and one number.")

        st.markdown("<br>", unsafe_allow_html=True)

        if auth_mode == "Login":
            if st.button("Sign In", key="login_btn", use_container_width=True):
                if not username.strip() or not password:
                    st.error("Please enter both username and password.")
                else:
                    success, msg = login_user(username.strip(), password)
                    if success:
                        st.session_state["logged_in"] = True
                        st.session_state["user_id"]   = username.strip().lower()
                        st.session_state["username"]  = username.strip().lower()
                        st.session_state["session_id"]= str(uuid.uuid4())[:8]
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        else:  # Register
            if st.button("Create Account", key="register_btn", use_container_width=True):
                if not username.strip() or not password:
                    st.error("Please fill in all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(username.strip(), password)
                    if success:
                        # Auto login after registration
                        st.session_state["logged_in"] = True
                        st.session_state["user_id"]   = username.strip().lower()
                        st.session_state["username"]  = username.strip().lower()
                        st.session_state["session_id"]= str(uuid.uuid4())[:8]
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    st.markdown(
        '<div style="text-align:center;margin-top:1rem;font-size:0.78rem;color:var(--text-muted);'
        'font-family:\'DM Mono\',monospace;">Passwords are hashed with bcrypt — never stored in plain text</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
<div style="max-width:420px;margin:1.2rem auto 0;background:#1a1520;border:1px solid #ff6b6b;
border-radius:10px;padding:1rem 1.2rem;text-align:left;">
<div style="font-size:0.78rem;color:#e8e4f0;line-height:1.7;">
⚠ <strong>This is not a replacement for professional therapy.</strong>
If you are in crisis, please call a helpline immediately.<br>
🇮🇳 iCall: <strong>9152987821</strong> &nbsp;·&nbsp;
🌍 <a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank"
style="color:#c8b8ff;">Find help near you</a>
</div>
</div>
""", unsafe_allow_html=True)


# ── Session ended screen ──────────────────────────────────────────────────────

def render_session_ended():
    turns = st.session_state["turn_count"]
    st.markdown(
        f'<div class="session-ended">'
        f'<div style="font-family:\'DM Serif Display\',serif;font-size:1.6rem;'
        f'color:var(--accent-2);margin-bottom:0.5rem;">Session Complete</div>'
        f'<div style="color:var(--text-muted);font-size:0.9rem;margin-bottom:1.5rem;">'
        f'{turns} turn{"s" if turns != 1 else ""} completed · Memory saved</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Show full chat history
    if st.session_state["chat_history"]:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("View full session transcript"):
            render_chat_history()

    # Start new session button
    st.markdown("<br>", unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        if st.button("Start New Session", key="new_session_btn", use_container_width=True):
            # Keep login but reset session
            uid      = st.session_state["user_id"]
            username = st.session_state["username"]
            facts    = st.session_state["confirmed_facts"]

            for key in list(st.session_state.keys()):
                del st.session_state[key]

            init_session()
            st.session_state["logged_in"] = True
            st.session_state["user_id"]   = uid
            st.session_state["username"]  = username
            st.session_state["confirmed_facts"] = facts
            st.session_state["session_id"]      = str(uuid.uuid4())[:8]
            st.rerun()


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    # ── Gate: login ──────────────────────────────────────────────────────────
    if not st.session_state["logged_in"]:
        render_login()
        return

    # ── Gate: session ended ───────────────────────────────────────────────────
    if st.session_state["session_ended"]:
        render_sidebar()
        render_session_ended()
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    render_sidebar()

    # ── Route to correct view ─────────────────────────────────────────────────
    current_view = st.session_state.get("current_view", "chat")

    if current_view == "user_stats":
        render_user_stats()
        return

    if current_view == "global_stats":
        render_global_stats()
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="cg-header">'
        '<h1 class="cg-title">Cognitive <span>Guardrail</span> AI</h1>'
        '<p class="cg-subtitle">CBT-informed · Self-auditing · Memory-aware</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Disclaimer banner ─────────────────────────────────────────────────────
    st.markdown("""
<div style="background:#1a1520;border:1px solid #ff6b6b;border-left:4px solid #ff6b6b;
border-radius:10px;padding:1rem 1.4rem;margin-bottom:1.5rem;">
<div style="font-family:'DM Mono',monospace;font-size:0.7rem;text-transform:uppercase;
letter-spacing:1.5px;color:#ff6b6b;margin-bottom:0.5rem;">⚠ Important Notice</div>
<div style="font-size:0.88rem;color:#e8e4f0;line-height:1.7;">
This tool is <strong>not a replacement for professional therapy or mental health treatment</strong>.
It is an educational AI assistant designed to help you recognize thought patterns.
If you are in distress or experiencing a mental health crisis, please reach out to a qualified professional.
</div>
<div style="margin-top:0.8rem;font-size:0.85rem;color:#e8e4f0;line-height:1.8;">
<strong style="color:#82e8c0;">Crisis Helplines:</strong><br>
🇮🇳 <strong>iCall (India):</strong> 9152987821 &nbsp;·&nbsp;
<strong>Vandrevala Foundation:</strong> 1860-2662-345 (24/7)<br>
🌍 <strong>International Association for Suicide Prevention:</strong>
<a href="https://www.iasp.info/resources/Crisis_Centres/" target="_blank"
style="color:#c8b8ff;">Find a crisis centre near you</a><br>
🇺🇸 <strong>988 Suicide & Crisis Lifeline (US):</strong> Call or text <strong>988</strong><br>
🇬🇧 <strong>Samaritans (UK):</strong> 116 123
</div>
</div>
""", unsafe_allow_html=True)

    # ── Chat history (previous turns) ─────────────────────────────────────────
    if st.session_state["chat_history"]:
        render_chat_history()
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────────────────────
    if not st.session_state["session_ended"]:
        placeholder = (
            "Continue sharing what's on your mind..."
            if st.session_state["turn_count"] > 0
            else "Share what you're thinking or feeling..."
        )

        user_input = st.text_area(
            "Input",
            placeholder      = placeholder,
            height           = 100,
            label_visibility = "collapsed",
            key              = f"input_{st.session_state['turn_count']}",
        )

        col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 4])

        with col_a:
            analyse_btn = st.button("Analyse", key="analyse_btn", use_container_width=True)
        with col_b:
            compare_btn = st.button("Compare ⇄", key="compare_btn", use_container_width=True)
        with col_c:
            end_btn = st.button("⏹ End Session", key="end_btn_main", use_container_width=True)
        with col_d:
            st.markdown(
                '<div style="padding:0.4rem 0;font-size:0.78rem;color:var(--text-muted);'
                'font-family:\'DM Mono\',monospace;line-height:1.5;">'
                '⇄ <b style="color:var(--accent-warm);">Compare</b> runs the same input '
                'twice — once with your memory context, once without — '
                'so you can see what the system learned about you over time.'
                '</div>',
                unsafe_allow_html=True,
            )

        # End session from main area
        if end_btn:
            st.session_state["session_ended"] = True
            st.rerun()

        # ── Run standard pipeline ─────────────────────────────────────────────
        if analyse_btn and user_input.strip():
            st.session_state["compare_mode"] = False
            with st.spinner("Thinking..."):
                result = run_full_pipeline(
                    user_input = user_input.strip(),
                    user_id    = st.session_state["user_id"],
                    session_id = st.session_state["session_id"],
                )
                st.session_state["last_result"]   = result
                st.session_state["last_result_nm"]= None
                st.session_state["turn_count"]   += 1
                st.session_state["chat_history"].append({
                    "user"      : user_input.strip(),
                    "response"  : result["response"],
                    "follow_up" : result["follow_up"],
                    "label"     : result["label"],
                    "technique" : result["technique"],
                })

        # ── Run compare pipeline ──────────────────────────────────────────────
        if compare_btn and user_input.strip():
            st.session_state["compare_mode"] = True
            with st.spinner("Running both pipelines..."):
                result_with    = run_full_pipeline(
                    user_input = user_input.strip(),
                    user_id    = st.session_state["user_id"],
                    session_id = st.session_state["session_id"],
                )
                result_without = run_pipeline_no_memory(user_input.strip())

                st.session_state["last_result"]    = result_with
                st.session_state["last_result_nm"] = result_without
                st.session_state["turn_count"]    += 1
                st.session_state["chat_history"].append({
                    "user"      : user_input.strip(),
                    "response"  : result_with["response"],
                    "follow_up" : result_with["follow_up"],
                    "label"     : result_with["label"],
                    "technique" : result_with["technique"],
                })

    # ── Render latest result ──────────────────────────────────────────────────
    if st.session_state["last_result"]:
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state["compare_mode"] and st.session_state["last_result_nm"]:
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                'text-transform:uppercase;letter-spacing:2px;color:var(--text-muted);'
                'margin-bottom:1rem;">Latest Turn — Comparison Mode</div>',
                unsafe_allow_html=True,
            )
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(
                    '<div class="compare-col-label with-memory">◈ With Memory</div>',
                    unsafe_allow_html=True,
                )
                render_result_card(st.session_state["last_result"], with_memory=True)

            with col_right:
                st.markdown(
                    '<div class="compare-col-label without-memory">◇ Without Memory</div>',
                    unsafe_allow_html=True,
                )
                render_result_card(st.session_state["last_result_nm"], with_memory=False)

            # Difference callout
            label_with    = st.session_state["last_result"]["label"]
            label_without = st.session_state["last_result_nm"]["label"]
            conf_diff     = round((st.session_state["last_result"]["confidence"] - st.session_state["last_result_nm"]["confidence"]) * 100)

            if label_with != label_without:
                st.markdown(
                    f'<div class="cg-card cg-card-warm" style="margin-top:1rem;">'
                    f'<div class="cg-label">Memory changed the classification</div>'
                    f'<div class="cg-value">Without memory: <b>{label_without}</b> → '
                    f'With memory: <b style="color:var(--accent);">{label_with}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                direction = "higher" if conf_diff > 0 else "lower"
                st.markdown(
                    f'<div class="cg-card cg-card-green" style="margin-top:1rem;">'
                    f'<div class="cg-label">Same label — memory affected confidence</div>'
                    f'<div class="cg-value">Confidence was {abs(conf_diff)}% {direction} with memory.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        else:
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                'text-transform:uppercase;letter-spacing:2px;color:var(--text-muted);'
                'margin-bottom:1rem;">Latest Turn Analysis</div>',
                unsafe_allow_html=True,
            )
            render_result_card(st.session_state["last_result"], with_memory=True)

            # AI self-audit
            ai_dist = st.session_state["last_result"].get("ai_distortions", [])
            if ai_dist:
                with st.expander(f"🔍 Memory self-audit — {len(ai_dist)} distortion(s) found in AI memory"):
                    for d in ai_dist:
                        st.markdown(
                            f'<div class="cg-card cg-card-warm">'
                            f'<div class="cg-label">{d.get("distortion","")}</div>'
                            f'<div class="cg-value" style="font-size:0.9rem;">'
                            f'<span style="color:var(--danger);">Found: </span>'
                            f'"{d.get("distorted_phrase","")}"<br>'
                            f'<span style="color:var(--accent-2);">Corrected: </span>'
                            f'{d.get("corrected","")}'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

        # ── Feedback buttons ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fb_col1, fb_col2, _ = st.columns([1, 1, 6])

        with fb_col1:
            if st.button("👍 Helpful", key=f"fb_good_{st.session_state['turn_count']}"):
                log_session_result(
                    user_id         = st.session_state["user_id"],
                    session_id      = st.session_state["session_id"],
                    analyst_label   = st.session_state["last_result"].get("label", ""),
                    da_verdict      = st.session_state["last_result"].get("da_verdict", ""),
                    da_suggested    = "",
                    consensus_loops = st.session_state["last_result"].get("consensus_loops", 0),
                    user_feedback   = "helpful",
                    flagged         = st.session_state["last_result"].get("flagged", False),
                    technique       = st.session_state["last_result"].get("technique", ""),
                )
                st.success("Thanks!")

        with fb_col2:
            if st.button("👎 Not helpful", key=f"fb_bad_{st.session_state['turn_count']}"):
                log_session_result(
                    user_id         = st.session_state["user_id"],
                    session_id      = st.session_state["session_id"],
                    analyst_label   = st.session_state["last_result"].get("label", ""),
                    da_verdict      = st.session_state["last_result"].get("da_verdict", ""),
                    da_suggested    = "",
                    consensus_loops = st.session_state["last_result"].get("consensus_loops", 0),
                    user_feedback   = "not helpful",
                    flagged         = st.session_state["last_result"].get("flagged", False),
                    technique       = st.session_state["last_result"].get("technique", ""),
                )
                st.warning("Feedback saved — system will self-adjust.")


if __name__ == "__main__":
    main()
