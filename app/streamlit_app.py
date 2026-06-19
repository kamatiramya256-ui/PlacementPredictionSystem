"""
streamlit_app.py
-----------------
Interactive dashboard for the AI-Powered Student Placement Prediction and
Career Guidance System.

Run locally:
    streamlit run app/streamlit_app.py

Features:
- Student information form
- Placement prediction + probability meter
- Skill gap analysis
- AI career guidance (Gemini API)
- Visual analytics dashboard
- Downloadable report
"""

import sys
import os
import pickle
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make utils/ importable whether the app is run from repo root or app/ dir
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from prediction import FEATURES, predict_placement, load_artifacts  # noqa: E402
from recommendations import analyze_skill_gap  # noqa: E402

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Placement Prediction & Career Guidance",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 2.2rem; font-weight: 700; margin-bottom: 0;}
    .sub-header {color: #666; margin-top: 0;}
    .metric-card {
        background: #f8f9fb; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #e6e6e6;
    }
    .gap-card {
        background: #fff7ed; border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.6rem;
    }
    .strength-pill {
        display: inline-block; background: #ecfdf5; color: #047857;
        border-radius: 999px; padding: 4px 12px; margin: 3px; font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🎓 AI-Powered Student Placement Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Placement probability, skill-gap analysis, and personalized AI career guidance</p>', unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------------------
# Load model artifacts (cached)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "placement_model.pkl")


@st.cache_resource
def get_model():
    try:
        artifacts = load_artifacts(MODEL_PATH)
        return artifacts["model"], artifacts["scaler"]
    except FileNotFoundError:
        return None, None


model, scaler = get_model()

if model is None:
    st.error(
        "Model file not found at `models/placement_model.pkl`. "
        "Run the training notebook (`notebooks/Model_Training.ipynb`) first to generate it."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: Student Information Form
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Student Information")

with st.sidebar.form("student_form"):
    name = st.text_input("Student Name", value="Student")
    cgpa = st.slider("CGPA", 4.0, 10.0, 7.2, 0.1)
    tenth = st.slider("10th Percentage", 40.0, 100.0, 78.0, 0.5)
    twelfth = st.slider("12th Percentage", 40.0, 100.0, 75.0, 0.5)
    internships = st.number_input("Number of Internships", 0, 10, 1)
    projects = st.number_input("Number of Projects", 0, 15, 2)
    coding = st.slider("Coding Skills Score", 0, 100, 60)
    communication = st.slider("Communication Skills Score", 0, 100, 65)
    aptitude = st.slider("Aptitude Score", 0, 100, 60)
    technical = st.slider("Technical Skills Score", 0, 100, 60)
    backlogs = st.number_input("Backlogs", 0, 10, 0)
    certifications = st.number_input("Certifications Count", 0, 15, 1)
    hackathons = st.number_input("Hackathon Participation", 0, 10, 1)
    use_gemini = st.checkbox("Generate AI career guidance (Gemini)", value=False)
    gemini_key_input = ""
    if use_gemini:
        gemini_key_input = st.text_input("Gemini API Key", type="password",
                                          help="Get a free key at https://aistudio.google.com/app/apikey")
    submitted = st.form_submit_button("🔮 Predict Placement", use_container_width=True)

student = {
    "CGPA": cgpa, "Tenth_Percentage": tenth, "Twelfth_Percentage": twelfth,
    "Internships": internships, "Projects": projects, "Coding_Skills": coding,
    "Communication_Skills": communication, "Aptitude_Score": aptitude,
    "Technical_Skills": technical, "Backlogs": backlogs,
    "Certifications": certifications, "Hackathon_Participation": hackathons,
}

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if not submitted and "last_result" not in st.session_state:
    st.info("👈 Fill in the student information form on the left and click **Predict Placement** to begin.")
    st.stop()

if submitted:
    label, prob_pct, confidence = predict_placement(model, scaler, student)
    gap_report = analyze_skill_gap(student)
    st.session_state["last_result"] = {
        "name": name, "student": student, "label": label,
        "prob_pct": prob_pct, "confidence": confidence, "gap_report": gap_report,
        "career_guidance": None,
    }

result = st.session_state["last_result"]
name = result["name"]
student = result["student"]
label = result["label"]
prob_pct = result["prob_pct"]
confidence = result["confidence"]
gap_report = result["gap_report"]

# --- Prediction summary -----------------------------------------------------
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.subheader("🎯 Placement Prediction")
    status_color = "#16a34a" if "Likely" in label else "#dc2626"
    st.markdown(
        f"""<div class="metric-card">
        <h3 style="color:{status_color}; margin:0;">{label}</h3>
        <p style="margin:0; color:#666;">for <b>{name}</b></p>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("📊 Probability")
    st.metric("Placement Probability", f"{prob_pct}%")

with col3:
    st.subheader("🔎 Confidence")
    st.metric("Confidence Level", confidence)

# --- Probability meter -------------------------------------------------------
st.markdown("#### Placement Probability Meter")
fig, ax = plt.subplots(figsize=(8, 1.2))
ax.barh([0], [100], color="#e5e7eb", height=0.5)
bar_color = "#16a34a" if prob_pct >= 50 else "#dc2626"
ax.barh([0], [prob_pct], color=bar_color, height=0.5)
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel("Probability (%)")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
st.pyplot(fig, use_container_width=True)

st.divider()

# --- Skill Gap Analysis ------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🧩 Skill Gap Analysis")
    for w in gap_report["weaknesses"]:
        st.markdown(
            f'<div class="gap-card"><b>{w["area"]}</b><br>{w["message"]}</div>',
            unsafe_allow_html=True,
        )

with right:
    st.subheader("💪 Strengths")
    if gap_report["strengths"]:
        pills = "".join(f'<span class="strength-pill">{s}</span>' for s in gap_report["strengths"])
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.write("No clear strength areas identified yet — focus on the gaps above.")

    st.subheader("📈 Profile Snapshot")
    profile_df = pd.DataFrame({
        "Metric": ["Coding", "Communication", "Aptitude", "Technical"],
        "Score": [student["Coding_Skills"], student["Communication_Skills"],
                  student["Aptitude_Score"], student["Technical_Skills"]],
    })
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=profile_df, x="Score", y="Metric", ax=ax2, palette="crest")
    ax2.set_xlim(0, 100)
    st.pyplot(fig2, use_container_width=True)

st.divider()

# --- AI Career Guidance (Gemini) --------------------------------------------
st.subheader("🤖 AI Career Guidance")

if use_gemini:
    if not gemini_key_input:
        st.warning("Enter your Gemini API key in the sidebar to generate AI career guidance.")
    else:
        if st.button("Generate Career Guidance", use_container_width=False):
            with st.spinner("Asking Gemini for personalized career guidance..."):
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=gemini_key_input)
                    gemini_model = genai.GenerativeModel("gemini-2.0-flash")

                    weaknesses_text = "\n".join(
                        f"- {w['area']}: {w['message']}" for w in gap_report["weaknesses"]
                    )
                    strengths_text = ", ".join(gap_report["strengths"]) or "None clearly identified"

                    prompt = f"""
You are an expert career counselor for engineering students. Based on the student profile
below, provide structured career guidance.

Student Profile: {student}
Placement Prediction: {label} ({prob_pct}% probability)
Identified Strengths: {strengths_text}
Identified Weaknesses:
{weaknesses_text}

Provide your response in clearly labeled sections:
1. Strengths Analysis
2. Weaknesses Analysis
3. Technologies to Learn (4-6 items)
4. Project Ideas (3-4 items, specific to their profile)
5. Recommended Certifications (3-4 items)
6. Interview Preparation Strategy (short actionable plan)
7. Suggested Career Paths (rank top 3 from: Software Engineer, Data Analyst, Data Scientist,
   Cloud Engineer, DevOps Engineer, Full Stack Developer, AI/ML Engineer)

Keep it concise, practical, and encouraging.
"""
                    response = gemini_model.generate_content(prompt)
                    st.session_state["last_result"]["career_guidance"] = response.text
                except Exception as e:
                    st.error(f"Could not reach Gemini API: {e}")

    if result.get("career_guidance"):
        st.markdown(result["career_guidance"])
else:
    st.caption("Enable 'Generate AI career guidance' in the sidebar form to get personalized "
               "Gemini-powered recommendations.")

st.divider()

# --- Visual Analytics Dashboard ----------------------------------------------
st.subheader("📊 Visual Analytics")
viz_tabs = st.tabs(["Profile Radar", "Score Comparison"])

with viz_tabs[0]:
    categories = ["CGPA", "Coding", "Communication", "Aptitude", "Technical", "Projects", "Internships"]
    raw_values = [
        student["CGPA"] * 10, student["Coding_Skills"], student["Communication_Skills"],
        student["Aptitude_Score"], student["Technical_Skills"],
        min(student["Projects"] * 10, 100), min(student["Internships"] * 25, 100),
    ]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    raw_values += raw_values[:1]
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax3.plot(angles, raw_values, color="#2563eb", linewidth=2)
    ax3.fill(angles, raw_values, color="#2563eb", alpha=0.25)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 100)
    st.pyplot(fig3, use_container_width=True)

with viz_tabs[1]:
    bench_df = pd.DataFrame({
        "Metric": ["CGPA (x10)", "Coding", "Communication", "Aptitude", "Technical"],
        "Student": [student["CGPA"] * 10, student["Coding_Skills"], student["Communication_Skills"],
                    student["Aptitude_Score"], student["Technical_Skills"]],
        "Recommended Benchmark": [75, 70, 65, 65, 65],
    })
    bench_melt = bench_df.melt(id_vars="Metric", var_name="Type", value_name="Score")
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=bench_melt, x="Metric", y="Score", hue="Type", ax=ax4, palette="Set2")
    ax4.set_ylim(0, 100)
    plt.xticks(rotation=20)
    st.pyplot(fig4, use_container_width=True)

st.divider()

# --- Downloadable report ------------------------------------------------------
st.subheader("📥 Download Report")


def build_report_text():
    lines = [
        "AI-Powered Student Placement Prediction Report",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Student Name: {name}",
        "",
        "Student Profile:",
    ]
    for k, v in student.items():
        lines.append(f"  - {k}: {v}")
    lines += [
        "",
        f"Placement Prediction: {label}",
        f"Placement Probability: {prob_pct}%",
        f"Confidence Level: {confidence}",
        "",
        "Skill Gap Analysis:",
    ]
    for w in gap_report["weaknesses"]:
        lines.append(f"  - [{w['area']}] {w['message']}")
    lines += ["", "Strengths: " + (", ".join(gap_report["strengths"]) or "None identified")]
    if result.get("career_guidance"):
        lines += ["", "AI Career Guidance:", "-" * 30, result["career_guidance"]]
    return "\n".join(lines)


report_text = build_report_text()
st.download_button(
    label="⬇️ Download Full Report (.txt)",
    data=report_text,
    file_name=f"placement_report_{name.replace(' ', '_')}.txt",
    mime="text/plain",
    use_container_width=True,
)

with st.expander("Preview report text"):
    st.text(report_text)
