"""
agent_tools.py
---------------
The agentic feature required by the capstone spec. Claude is given
two tools and autonomously decides when to call them while coaching
the user:

  1. fetch_job_posting(url)   - pull a live job description from a URL
                                 the user pasted, instead of requiring
                                 them to copy/paste the text manually
  2. lookup_market_skills(role) - look up in-demand skills/keywords for
                                 a role, so advice reflects the current
                                 market rather than only the one job ad

Both tools are plain Python functions with an Anthropic tool schema.
The agent loop that wires them together lives in pipeline.py.
"""

from __future__ import annotations
import re
import os
import requests

TOOL_SCHEMAS = [
    {
        "name": "fetch_job_posting",
        "description": (
            "Fetch and extract the plain-text job description from a "
            "public job-posting URL the user provided. Use this when the "
            "user pastes a link instead of the job description text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The job posting URL"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "lookup_market_skills",
        "description": (
            "Look up currently in-demand skills and keywords for a given "
            "job title/role, to check the resume advice against real "
            "market signal rather than a single job posting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "Job title, e.g. 'data analyst'"}
            },
            "required": ["role"],
        },
    },
]


def _to_openai_tool(schema: dict) -> dict:
    """Convert an Anthropic-style tool schema (name/description/input_schema)
    into the OpenAI/Groq function-calling format (nested under 'function')."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["input_schema"],
        },
    }


# Groq's API is OpenAI-compatible, so tool schemas need the OpenAI shape
# rather than Anthropic's flatter shape. Kept in sync with TOOL_SCHEMAS above.
OPENAI_TOOL_SCHEMAS = [_to_openai_tool(s) for s in TOOL_SCHEMAS]


def fetch_job_posting(url: str, timeout: int = 10) -> str:
    """Fetch a URL and return a crude text-only extraction of the body.
    Kept dependency-free (no bs4) with a simple tag stripper, which is
    sufficient for a capstone demo; swap in a proper HTML parser for
    production use."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:8000]  # cap length to keep prompts manageable
    except Exception as exc:
        return f"[TOOL ERROR] Could not fetch {url}: {exc}"


# A tiny local knowledge base used when no SERPAPI_API_KEY is configured,
# so the agentic feature still works offline/mock for grading.
_MOCK_MARKET_SKILLS = {
    "data analyst": ["SQL", "Python", "Power BI/Tableau", "A/B testing", "statistics"],
    "software engineer": ["System design", "CI/CD", "cloud (AWS/GCP/Azure)", "testing", "Git"],
    "product manager": ["Roadmapping", "stakeholder communication", "SQL", "user research", "A/B testing"],
    "career coach": ["Resume writing", "interview coaching", "LinkedIn optimization", "career planning"],
    "default": ["Communication", "problem solving", "domain-specific tools", "collaboration"],
}


def lookup_market_skills(role: str) -> str:
    api_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if api_key:
        try:
            resp = requests.get(
                "https://serpapi.com/search",
                params={"engine": "google", "q": f"{role} in-demand skills 2026", "api_key": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            snippets = [r.get("snippet", "") for r in data.get("organic_results", [])[:5]]
            return " ".join(snippets) or "No market data found."
        except Exception as exc:
            return f"[TOOL ERROR] Market lookup failed: {exc}"

    key = role.lower().strip()
    for known_role, skills in _MOCK_MARKET_SKILLS.items():
        if known_role in key:
            return f"[MOCK MARKET DATA] Commonly requested skills for {role}: {', '.join(skills)}."
    return f"[MOCK MARKET DATA] Commonly requested skills for {role}: {', '.join(_MOCK_MARKET_SKILLS['default'])}."


def run_tool(name: str, tool_input: dict) -> str:
    if name == "fetch_job_posting":
        return fetch_job_posting(tool_input["url"])
    if name == "lookup_market_skills":
        return lookup_market_skills(tool_input["role"])
    return f"[TOOL ERROR] Unknown tool: {name}"
