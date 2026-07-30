"""
app.py
-------
Streamlit front-end for the Resume Analyzer & Career Coach capstone project.

Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src import resume_parser
from src.pipeline import CareerCoachPipeline

load_dotenv()

st.set_page_config(page_title="Resume Analyzer & Career Coach", page_icon="🧭", layout="wide")

APP_MODE = os.environ.get("APP_MODE", "mock").lower()

st.title("🧭 Resume Analyzer & Career Coach")
st.caption(
    "AI-powered capstone project — Groq (Llama) dual-model analysis, "
    "RAG-based resume/job matching, and an agentic job-URL fetcher."
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "App mode",
        options=["mock", "live"],
        index=0 if APP_MODE == "mock" else 1,
        help=(
            "mock = no API calls, canned demo responses (safe for grading/demo). "
            "live = real Groq (Llama) calls, requires GROQ_API_KEY in .env. "
            "OPENAI_API_KEY is optional (adds a real GPT cross-check + real embeddings)."
        ),
    )
    if mode == "live":
        if not os.environ.get("GROQ_API_KEY"):
            st.warning("Add GROQ_API_KEY to your .env to use live mode.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.info("No OPENAI_API_KEY set — cross-check will use a second Groq model, and RAG will use local embeddings instead of OpenAI's.")
    st.divider()
    role_hint = st.text_input("Target role (optional)", placeholder="e.g. Data Analyst")
    st.divider()
    st.caption("See docs/ARCHITECTURE.md and docs/PROMPTS.md for full technical documentation.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Your resume")
    resume_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    resume_text_input = st.text_area("...or paste resume text", height=200)

with col2:
    st.subheader("2. Job description")
    job_input = st.text_area(
        "Paste the job description OR a job posting URL",
        height=200,
        placeholder="Paste text, or paste a URL like https://company.com/careers/123",
    )

run_clicked = st.button("Analyze fit", type="primary")

if run_clicked:
    # Resolve resume text from file or textarea
    resume_text = ""
    if resume_file is not None:
        resume_text = resume_parser.extract_text(resume_file.name, resume_file.read())
    elif resume_text_input.strip():
        resume_text = resume_parser.clean_text(resume_text_input)

    if not resume_text.strip():
        st.error("Please upload or paste a resume.")
    elif not job_input.strip():
        st.error("Please paste a job description or job posting URL.")
    else:
        with st.spinner("Running RAG retrieval + dual-model analysis..."):
            pipeline = CareerCoachPipeline(mode=mode)
            result = pipeline.run(resume_text, job_input, role_hint=role_hint)

        if result.blocked:
            st.error("This submission was blocked by the safety guardrails:")
            for note in result.guardrail_notes:
                st.write(f"- {note}")
        else:
            st.success("Analysis complete.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Keyword match score", f"{result.match_score}%")
            m2.metric("Matched keywords", len(result.matched_keywords))
            m3.metric("Missing keywords", len(result.missing_keywords))

            if result.matched_keywords:
                st.write("**Matched:** " + ", ".join(result.matched_keywords))
            if result.missing_keywords:
                st.write("**Missing:** " + ", ".join(result.missing_keywords))

            if result.tool_calls_made:
                st.info(f"Agent tools used: {', '.join(result.tool_calls_made)}")

            if result.recommended_certifications:
                st.subheader("🎓 Recommended certifications")
                st.caption(
                    "Based on the skill gaps and matches found above — closing "
                    "these gaps first will move the match score the most."
                )
                for cert in result.recommended_certifications:
                    tag = "🔴 Missing skill" if cert.reason == "missing" else "🟡 Strengthen existing skill"
                    st.markdown(
                        f"- **[{cert.name}]({cert.link})** — *{cert.provider}*  \n"
                        f"  {tag}: `{cert.skill}`"
                    )

            tab1, tab2 = st.tabs(["🟣 Primary analysis (Llama 3.3 70B)", "🟢 Cross-check analysis"])
            with tab1:
                st.markdown(result.claude_analysis)
            with tab2:
                st.markdown(result.openai_analysis)

            if result.guardrail_notes:
                with st.expander("Guardrail notes (non-blocking)"):
                    for note in result.guardrail_notes:
                        st.write(f"- {note}")

st.divider()
st.caption(
    "Built for the End-to-End AI-Powered Application capstone. "
    "See README.md for setup, docs/ARCHITECTURE.md for the system diagram, "
    "and evaluation/EVALUATION_REPORT.md for test results."
)
