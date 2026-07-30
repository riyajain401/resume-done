Deployed link:- https://resumecareercoach.streamlit.app/



# Resume Analyzer & Career Coach

An end-to-end AI-powered capstone project: upload a resume and a job
description (or job URL), and get a RAG-grounded, dual-model analysis
of how well you match — plus concrete, specific improvements.

Built to satisfy the capstone's technical requirements:

| Requirement | How it's met |
|---|---|
| 2+ LLM integrations | Groq/Llama 3.3 70B (primary reasoning + tool-calling agent) and a second model (real OpenAI GPT if `OPENAI_API_KEY` is set, otherwise Groq/Llama 3.1 8B as an independent cross-check) — see `src/llm_clients.py` |
| RAG pipeline with vector database | Resume is chunked, embedded, and indexed in FAISS; retrieval is run against the job description before either LLM sees the resume — see `src/rag_engine.py` |
| Agentic feature | Claude can autonomously call `fetch_job_posting(url)` and `lookup_market_skills(role)` mid-reasoning — see `src/agent_tools.py` |
| Safety guardrails + moderation | PII scrubbing, prompt-injection detection, and content moderation on both input and output — see `src/guardrails.py` |
| Evaluation report | 6 automated test cases covering match accuracy and safety behavior — see `evaluation/EVALUATION_REPORT.md` |
| Deployed web interface | Streamlit app — `app.py` |
| GitHub repo, README, documented prompts | This repo — see `docs/PROMPTS.md` for prompt documentation |

## Quick start

```bash
git clone <your-repo-url>
cd resume-career-coach
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GROQ_API_KEY for live mode (free at console.groq.com/keys),
# or leave APP_MODE=mock to demo without any keys.
# OPENAI_API_KEY is optional: adds a real GPT cross-check + real embeddings.

streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Demo without API keys

The app ships with `APP_MODE=mock` by default. In this mode:
- The RAG pipeline (chunking, embedding, FAISS retrieval) runs for real,
  using a deterministic hashing-based embedder instead of a paid API.
- The keyword match score is computed for real (it never depends on an LLM).
- Claude/GPT responses are replaced with clearly-labeled mock text, so
  you can see the full UI and flow without spending API credits.

Switch to **live** mode in the sidebar (with real keys in `.env`) for
actual model-generated coaching advice and agent tool use.

## Project structure

```
resume-career-coach/
├── app.py                     # Streamlit UI
├── requirements.txt
├── .env.example
├── src/
│   ├── resume_parser.py       # PDF/DOCX/TXT text extraction + chunking
│   ├── guardrails.py          # PII scrub, prompt-injection scan, moderation
│   ├── rag_engine.py          # Embeddings + FAISS vector store
│   ├── agent_tools.py         # Agentic tools (job URL fetch, market lookup)
│   ├── llm_clients.py         # Claude + OpenAI wrappers (with mock mode)
│   └── pipeline.py            # Orchestrates the full analysis pipeline
├── evaluation/
│   ├── test_cases.json        # 6 test cases (match accuracy + safety)
│   ├── evaluate.py            # Test harness -> writes EVALUATION_REPORT.md
│   └── EVALUATION_REPORT.md   # Latest results (100% pass in mock mode)
├── docs/
│   ├── ARCHITECTURE.md        # System diagram + design rationale
│   └── PROMPTS.md             # Documented system/user prompts
└── data/
    ├── sample_resumes/        # Sample resumes used by the test cases
    └── sample_jobs/           # Sample job descriptions used by the test cases
```

## Running the evaluation suite

```bash
python -m evaluation.evaluate
```

This runs all 6 test cases (2 strong matches, 2 mismatches, 1 prompt-injection
probe, 1 PII-scrubbing probe) and regenerates `evaluation/EVALUATION_REPORT.md`.
It defaults to `APP_MODE=mock` so it can run in CI without API keys; set
`APP_MODE=live` to also evaluate real LLM output quality.

## Known limitations / next steps

- The keyword vocabulary (`SKILL_VOCAB` in `src/pipeline.py`) is a small,
  readable list for demo purposes — a production version would use a
  much larger skills taxonomy.
- `fetch_job_posting` does simple tag-stripping instead of a full HTML
  parser; sites with heavy client-side rendering may not extract cleanly.
- The FAISS index is in-memory and per-session; a production deployment
  would persist embeddings for repeat users.
