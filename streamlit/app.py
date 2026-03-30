import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Predicting Burnout Risk", page_icon="🔥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at 20% 20%, #0a0a0a, #000000 70%);
    color: #eaeaea;
}

body::before {
    content: "";
    position: fixed;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(255,65,108,0.15), transparent 70%);
    top: -100px;
    left: -100px;
    filter: blur(80px);
    z-index: -1;
}

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 28px;
    margin-bottom: 20px;
    transition: 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.gradient-text {
    background: linear-gradient(90deg, #FF4B2B, #FF416C, #ff9a9e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
}

.subtitle {
    color: #9a9a9a;
    font-size: 1.1rem;
}

.stButton>button {
    background: linear-gradient(90deg, #FF4B2B, #FF416C);
    color: white;
    border: none;
    padding: 14px;
    border-radius: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.25s ease;
    width: 100%;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 10px 30px rgba(255, 65, 108, 0.4);
}

.empty-box {
    height: 550px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #555;
    text-align: center;
}

header, #MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:-40px;">
    <h1 class="gradient-text">🔥 Predicting Burnout Risk</h1>
    <p class="subtitle">Monitoring employee well-being to boost engagement and performance.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("### ⚙️ Parameters")

    work_hours = st.slider("Daily Work Hours", 0.0, 16.0, 8.0)
    sleep_hours = st.slider("Sleep Duration", 0.0, 16.0, 8.0)
    
    meetings_count = st.slider("Meetings Count", 0, 20, 3)

    c1, c2 = st.columns(2)
    with c1:
        fatigue = st.select_slider("Fatigue", options=list(range(1, 11)), value=5)
    with c2:
        isolation = st.select_slider("Isolation", options=list(range(1, 11)), value=3)

    screen_time = st.slider("Screen Time", 0.0, 16.0, 8.0)
    task_completion = st.slider("Task Completion %", 0, 100, 80)

    with st.expander("🛠️ Advanced Parameters (Optional)"):
        day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
        after_hours_work = st.selectbox("After Hours Work", ["Yes", "No"])
        breaks_taken = st.number_input("Breaks Taken", min_value=0, value=10)
        app_switches = st.number_input("App Switches ", min_value=0, value=50)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict")

with col_right:
    if not predict_btn:
        st.markdown("""
        <div class="glass-card empty-box">
            <h3>🧠 Ready for Analysis</h3>
            <p>Adjust your employee lifestyle metrics and click "🚀 Predict".</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Analyzing mental state..."):
            import time
            time.sleep(1.2)

        score = (work_hours * 1.1) + (fatigue * 1.4) + (meetings_count * 0.4) - (sleep_hours * 0.7)

        if score > 15:
            status, color = "HIGH", "#FF416C"
            msg = "High risk of burnout. Immediate rest recommended."
        elif score > 10:
            status, color = "MEDIUM", "#FFB75E"
            msg = "Moderate stress detected. Balance workload."
        else:
            status, color = "LOW", "#00F260"
            msg = "Great balance! Keep maintaining this routine."

        st.markdown(f"""
        <div class="glass-card" style="
            border-left: 6px solid {color};
            box-shadow: 0 0 40px {color}40;
        ">
            <p style="color:#888; letter-spacing:2px; font-size:0.75rem;">BURNOUT STATUS</p>
            <h1 style="color:{color}; margin:10px 0; font-size:2.8rem;">{status}</h1>
            <p style="color:#bbb; font-size:1.05rem;">{msg}</p>
        </div>
        """, unsafe_allow_html=True)

        categories = ['Workload', 'Rest', 'Social', 'Energy', 'Efficiency']
        values = [
            work_hours / 1.6,
            sleep_hours,
            10 - isolation,
            fatigue,
            task_completion / 10
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}",
            line=dict(color=color, width=4)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 10]),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#888")),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=30),
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align:center; padding:20px; color:#444; font-size:0.8rem;">
    © 2026 Burnout Prediction Systems
</div>
""", unsafe_allow_html=True)