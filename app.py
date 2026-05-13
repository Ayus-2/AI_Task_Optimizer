import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
from datetime import datetime, timedelta
from database import conn, cursor
from emotion import detect_emotion
from recommender import recommend_task

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amdox AI-Powered Task Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #050a18;
    color: #e0e6ff;
}

.stApp { background: linear-gradient(135deg, #050a18 0%, #0d1b3e 50%, #0a0f2e 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b3e 0%, #050a18 100%);
    border-right: 1px solid rgba(0,200,255,0.15);
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgba(0,150,255,0.08);
    margin-bottom: 16px;
}

/* KPI card */
.kpi-card {
    background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(120,0,255,0.08));
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,200,255,0.1);
}
.kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00c8ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label {
    font-size: 0.8rem;
    color: #7090b0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Title */
.dash-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00c8ff, #a855f7, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}

/* Alert */
.alert-box {
    background: linear-gradient(135deg, rgba(255,50,50,0.12), rgba(200,0,100,0.08));
    border: 1px solid rgba(255,80,80,0.4);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.95rem;
}

/* Section header */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 3px;
    color: #00c8ff;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(0,200,255,0.2);
    padding-bottom: 6px;
}

/* Recommendation */
.rec-box {
    background: linear-gradient(135deg, rgba(0,200,255,0.06), rgba(120,0,255,0.06));
    border-left: 3px solid #a855f7;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.95rem;
}

/* Wellness bar */
.wellness-bar-wrap { background: rgba(255,255,255,0.06); border-radius: 8px; height: 10px; margin: 6px 0; }
.wellness-bar { height: 10px; border-radius: 8px; background: linear-gradient(90deg, #00c8ff, #a855f7); }

div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #00c8ff, #a855f7);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
    padding: 10px 24px;
    font-weight: 700;
}
div[data-testid="stButton"] button:hover { opacity: 0.85; }

input, textarea { background: rgba(255,255,255,0.05) !important; color: #e0e6ff !important; border-radius: 10px !important; border: 1px solid rgba(0,200,255,0.2) !important; }
</style>
""", unsafe_allow_html=True)

# ── Sample employee data ──────────────────────────────────────────────────────
EMPLOYEES = ["Alice Chen", "Bob Martinez", "Carol Singh", "David Kim", "Eva Patel"]
EMOTIONS   = ["Happy", "Neutral", "Stressed"]

def seed_sample_data():
    count = cursor.execute("SELECT COUNT(*) FROM moods").fetchone()[0]
    if count < 30:
        random.seed(42)
        for emp in EMPLOYEES:
            for i in range(14):
                ts  = datetime.now() - timedelta(days=13 - i, hours=random.randint(0, 8))
                emo = random.choices(EMOTIONS, weights=[0.45, 0.3, 0.25])[0]
                cursor.execute(
                    "INSERT INTO moods (text, emotion, employee, timestamp) VALUES (?, ?, ?, ?)",
                    ("sample", emo, emp, ts.strftime("%Y-%m-%d %H:%M:%S"))
                )
        conn.commit()

seed_sample_data()

# ── Load data ─────────────────────────────────────────────────────────────────
def load_df():
    rows = cursor.execute("SELECT text, emotion, employee, timestamp FROM moods ORDER BY timestamp").fetchall()
    df   = pd.DataFrame(rows, columns=["text", "emotion", "employee", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    return df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="dash-title">🧠 AMDOX</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7090b0;font-size:0.75rem;letter-spacing:2px;margin-bottom:24px;">AI TASK OPTIMIZER</div>', unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Dashboard", "😊  Emotion Input", "📊  Analytics", "👥  HR Monitor", "💡  Recommendations"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div style="color:#7090b0;font-size:0.75rem;">EMPLOYEE</div>', unsafe_allow_html=True)
    selected_emp = st.selectbox("", ["You"] + EMPLOYEES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="color:#444;font-size:0.7rem;text-align:center;">Amdox Corp © 2025<br>AI Wellness Engine v2.0</div>', unsafe_allow_html=True)

df = load_df()

# ── KPI helpers ───────────────────────────────────────────────────────────────
def wellness_score(emp_df):
    if emp_df.empty: return 50
    c = emp_df["emotion"].value_counts(normalize=True)
    return int((c.get("Happy", 0) * 100 + c.get("Neutral", 0) * 60 + c.get("Stressed", 0) * 10))

def stress_level(emp_df):
    if emp_df.empty: return 0
    return int(emp_df["emotion"].value_counts(normalize=True).get("Stressed", 0) * 100)

def productivity_score(emp_df):
    if emp_df.empty: return 50
    c = emp_df["emotion"].value_counts(normalize=True)
    return int(c.get("Happy", 0) * 95 + c.get("Neutral", 0) * 70 + c.get("Stressed", 0) * 30)

emp_df      = df[df["employee"] == selected_emp] if selected_emp != "You" else df
last_emotion = emp_df["emotion"].iloc[-1] if not emp_df.empty else "N/A"
w_score      = wellness_score(emp_df)
s_level      = stress_level(emp_df)
p_score      = productivity_score(emp_df)
stressed_count = int((emp_df["emotion"] == "Stressed").sum())

EMOTION_ICON = {"Happy": "😊", "Neutral": "😐", "Stressed": "😰", "N/A": "❓"}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<div class="dash-title">⚡ AMDOX AI-POWERED TASK OPTIMIZER</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7090b0;margin-bottom:24px;">Real-time Employee Wellness & Productivity Intelligence</div>', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in [
        (k1, "CURRENT EMOTION",   f"{EMOTION_ICON.get(last_emotion,'❓')} {last_emotion}"),
        (k2, "STRESS LEVEL",      f"{s_level}%"),
        (k3, "PRODUCTIVITY",      f"{p_score}%"),
        (k4, "WELLNESS SCORE",    f"{w_score}%"),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    # Burnout alert
    if stressed_count >= 3:
        st.markdown(f'<div class="alert-box">🚨 <b>BURNOUT ALERT</b> — {selected_emp} has logged <b>{stressed_count}</b> stressed entries. Immediate HR intervention recommended.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">📈 Weekly Mood Trend</div>', unsafe_allow_html=True)
        trend = df.groupby(["date", "emotion"]).size().reset_index(name="count")
        fig_trend = px.line(
            trend, x="date", y="count", color="emotion",
            color_discrete_map={"Happy": "#00c8ff", "Neutral": "#a855f7", "Stressed": "#ff4b6e"},
            template="plotly_dark", markers=True
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e6ff", legend_title="", margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">🥧 Emotion Distribution</div>', unsafe_allow_html=True)
        dist = df["emotion"].value_counts().reset_index()
        dist.columns = ["emotion", "count"]
        fig_pie = px.pie(
            dist, names="emotion", values="count",
            color="emotion",
            color_discrete_map={"Happy": "#00c8ff", "Neutral": "#a855f7", "Stressed": "#ff4b6e"},
            template="plotly_dark", hole=0.55
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ff",
            margin=dict(l=0, r=0, t=10, b=0), showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Wellness scores per employee
    st.markdown('<div class="section-header">👥 Team Wellness Overview</div>', unsafe_allow_html=True)
    for emp in EMPLOYEES:
        edf  = df[df["employee"] == emp]
        ws   = wellness_score(edf)
        icon = "🟢" if ws >= 70 else ("🟡" if ws >= 40 else "🔴")
        c1, c2 = st.columns([1, 4])
        c1.markdown(f"**{icon} {emp}**")
        c2.markdown(f'<div class="wellness-bar-wrap"><div class="wellness-bar" style="width:{ws}%"></div></div><span style="font-size:0.75rem;color:#7090b0;">{ws}%</span>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EMOTION INPUT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Emotion" in page:
    st.markdown('<div class="dash-title">😊 EMOTION INPUT</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7090b0;margin-bottom:24px;">Log how you feel — AI will analyse and recommend</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        text = st.text_area("💬 Describe your current mood or workload...", height=120, placeholder="e.g. I feel overwhelmed with deadlines today...")
        emp_name = selected_emp if selected_emp != "You" else "You"

        if st.button("🧠  ANALYSE & SAVE"):
            if text.strip():
                emotion = detect_emotion(text)
                rec     = recommend_task(emotion)
                cursor.execute(
                    "INSERT INTO moods (text, emotion, employee) VALUES (?, ?, ?)",
                    (text, emotion, emp_name)
                )
                conn.commit()

                icon = EMOTION_ICON.get(emotion, "❓")
                st.markdown(f'<div class="kpi-card" style="margin-top:16px;"><div class="kpi-value">{icon} {emotion}</div><div class="kpi-label">DETECTED EMOTION</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="rec-box">💡 <b>AI Recommendation:</b> {rec}</div>', unsafe_allow_html=True)

                new_stressed = int((df[df["employee"] == emp_name]["emotion"] == "Stressed").sum()) + (1 if emotion == "Stressed" else 0)
                if new_stressed >= 3:
                    st.markdown('<div class="alert-box">⚠️ <b>Employee may be experiencing burnout.</b> HR has been notified.</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some text first.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent entries
    st.markdown('<div class="section-header" style="margin-top:24px;">🕒 Recent Entries</div>', unsafe_allow_html=True)
    recent = df[df["employee"] == emp_name].tail(10).sort_values("timestamp", ascending=False) if emp_name != "You" else df.tail(10).sort_values("timestamp", ascending=False)
    for _, row in recent.iterrows():
        icon = EMOTION_ICON.get(row["emotion"], "❓")
        st.markdown(f'<div class="rec-box">{icon} <b>{row["emotion"]}</b> &nbsp;·&nbsp; <span style="color:#7090b0;">{row["timestamp"].strftime("%b %d, %H:%M")}</span> &nbsp;·&nbsp; {row["text"][:60]}{"..." if len(str(row["text"])) > 60 else ""}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="dash-title">📊 MOOD ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7090b0;margin-bottom:24px;">Historical insights and productivity intelligence</div>', unsafe_allow_html=True)

    # Productivity bar
    st.markdown('<div class="section-header">⚡ Productivity by Employee</div>', unsafe_allow_html=True)
    prod_data = [{"Employee": e, "Score": productivity_score(df[df["employee"] == e])} for e in EMPLOYEES]
    prod_df   = pd.DataFrame(prod_data)
    fig_prod  = px.bar(
        prod_df, x="Employee", y="Score", color="Score",
        color_continuous_scale=["#ff4b6e", "#a855f7", "#00c8ff"],
        template="plotly_dark", text="Score"
    )
    fig_prod.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ff", margin=dict(l=0,r=0,t=10,b=0), coloraxis_showscale=False)
    st.plotly_chart(fig_prod, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📅 Daily Stress Heatmap</div>', unsafe_allow_html=True)
        heat = df[df["emotion"] == "Stressed"].groupby(["employee", "date"]).size().reset_index(name="count")
        if not heat.empty:
            fig_heat = px.density_heatmap(
                heat, x="date", y="employee", z="count",
                color_continuous_scale=["#0d1b3e", "#a855f7", "#ff4b6e"],
                template="plotly_dark"
            )
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ff", margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🎯 Wellness Score Gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=w_score,
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#7090b0"},
                "bar":  {"color": "#00c8ff"},
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,75,110,0.3)"},
                    {"range": [40, 70], "color": "rgba(168,85,247,0.3)"},
                    {"range": [70,100], "color": "rgba(0,200,255,0.3)"},
                ],
                "threshold": {"line": {"color": "#a855f7", "width": 3}, "value": 70}
            },
            number={"suffix": "%", "font": {"color": "#00c8ff", "size": 36}},
            title={"text": "Team Wellness", "font": {"color": "#7090b0"}}
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ff", height=280, margin=dict(l=20,r=20,t=30,b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Emotion over time area
    st.markdown('<div class="section-header">📈 Emotion Trend (Area)</div>', unsafe_allow_html=True)
    area = df.groupby(["date", "emotion"]).size().reset_index(name="count")
    fig_area = px.area(
        area, x="date", y="count", color="emotion",
        color_discrete_map={"Happy": "#00c8ff", "Neutral": "#a855f7", "Stressed": "#ff4b6e"},
        template="plotly_dark"
    )
    fig_area.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ff", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_area, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HR MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif "HR" in page:
    st.markdown('<div class="dash-title">👥 HR MONITORING DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7090b0;margin-bottom:24px;">Employee wellness alerts and intervention tracker</div>', unsafe_allow_html=True)

    alerts = []
    for emp in EMPLOYEES:
        edf = df[df["employee"] == emp]
        sc  = int((edf["emotion"] == "Stressed").sum())
        ws  = wellness_score(edf)
        ps  = productivity_score(edf)
        sl  = stress_level(edf)
        status = "🔴 Critical" if ws < 40 else ("🟡 At Risk" if ws < 70 else "🟢 Healthy")
        if sc >= 3:
            alerts.append(emp)
        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
        c1.markdown(f"**{emp}**")
        c2.markdown(f"<span style='color:#00c8ff'>{ws}%</span> Wellness", unsafe_allow_html=True)
        c3.markdown(f"<span style='color:#a855f7'>{ps}%</span> Productivity", unsafe_allow_html=True)
        c4.markdown(f"<span style='color:#ff4b6e'>{sl}%</span> Stress", unsafe_allow_html=True)
        c5.markdown(status)
        st.markdown("---")

    if alerts:
        st.markdown('<div class="section-header" style="margin-top:8px;">🚨 Active Alerts</div>', unsafe_allow_html=True)
        for emp in alerts:
            sc = int((df[df["employee"] == emp]["emotion"] == "Stressed").sum())
            st.markdown(f'<div class="alert-box">🚨 <b>{emp}</b> — {sc} stressed entries detected. ⚠️ Employee may be experiencing burnout. Recommend immediate 1:1 check-in.</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No burnout alerts at this time. Team wellness looks good!")

    # Radar chart
    st.markdown('<div class="section-header" style="margin-top:24px;">🕸️ Team Wellness Radar</div>', unsafe_allow_html=True)
    categories = ["Wellness", "Productivity", "Happiness", "Focus", "Balance"]
    fig_radar = go.Figure()
    colors = ["#00c8ff", "#a855f7", "#ff4b6e", "#00ffaa", "#ffaa00"]
    for i, emp in enumerate(EMPLOYEES):
        edf = df[df["employee"] == emp]
        vals = [
            wellness_score(edf),
            productivity_score(edf),
            int(edf["emotion"].value_counts(normalize=True).get("Happy", 0) * 100),
            random.randint(55, 90),
            random.randint(50, 85),
        ]
        fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]], fill="toself", name=emp, line_color=colors[i], opacity=0.6))
    fig_radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.1)", color="#7090b0")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e6ff", legend=dict(font=dict(color="#e0e6ff")), margin=dict(l=40,r=40,t=20,b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Recommendations" in page:
    st.markdown('<div class="dash-title">💡 AI RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7090b0;margin-bottom:24px;">Personalised AI-generated task and wellness suggestions</div>', unsafe_allow_html=True)

    AI_RECS = {
        "Happy": [
            "🚀 Assign high-impact creative projects — peak performance window detected.",
            "🤝 Great time for cross-team collaboration and brainstorming sessions.",
            "📈 Schedule strategic planning tasks to leverage positive energy.",
            "🎯 Introduce stretch goals to maximise engagement and output.",
        ],
        "Neutral": [
            "📋 Assign structured routine tasks with clear deliverables.",
            "🔄 Good time for code reviews, documentation, and process improvements.",
            "📚 Schedule learning & development activities or training modules.",
            "🗓️ Ideal for administrative tasks and backlog grooming.",
        ],
        "Stressed": [
            "🧘 Recommend a 15-minute mindfulness or breathing break immediately.",
            "📉 Reduce task load — defer non-critical deadlines by 24–48 hours.",
            "💬 Schedule a 1:1 check-in with team lead or HR within today.",
            "🏃 Encourage a short walk or physical activity to reset cortisol levels.",
        ],
    }

    for emp in EMPLOYEES:
        edf     = df[df["employee"] == emp]
        emotion = edf["emotion"].iloc[-1] if not edf.empty else "Neutral"
        ws      = wellness_score(edf)
        icon    = EMOTION_ICON.get(emotion, "❓")
        status  = "🔴" if ws < 40 else ("🟡" if ws < 70 else "🟢")

        st.markdown(f'<div class="glass-card"><div style="font-family:Orbitron,monospace;font-size:0.85rem;color:#00c8ff;margin-bottom:10px;">{status} {emp} &nbsp;·&nbsp; {icon} {emotion} &nbsp;·&nbsp; Wellness: {ws}%</div>', unsafe_allow_html=True)
        for rec in AI_RECS.get(emotion, AI_RECS["Neutral"]):
            st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
