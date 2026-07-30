"""
llm_clients.py
---------------
Wraps the two LLM integrations required by the capstone spec:

  - Groq (Llama 3.3 70B)   -> primary reasoning model (resume critique,
                              coaching narrative, tool-calling agent)
  - Groq (Llama 3.1 8B)    -> secondary model, used to cross-check /
                              independently re-score the match, so the
                              user sees where two models agree or disagree
                              (automatically swaps to real OpenAI GPT
                              instead, if OPENAI_API_KEY is provided)

Groq's API is OpenAI-compatible, so we reuse the `openai` Python SDK
for both Groq calls and (optional) real OpenAI calls — just pointed at
a different base_url and key.

Both roles support a "mock" mode that returns canned, clearly labeled
responses, so the whole app is demoable without API keys or network
access (useful for grading / offline demos).
"""

from __future__ import annotations
import os

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_PRIMARY_MODEL = "llama-3.3-70b-versatile"
GROQ_SECONDARY_MODEL = "llama-3.1-8b-instant"


class LLMClients:
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self.primary_client = None      # Groq client (primary reasoning + agent)
        self.secondary_client = None    # OpenAI client if key present, else Groq again
        self.secondary_model = None
        self.secondary_is_openai = False

        if self.mode == "live":
            import openai

            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise RuntimeError(
                    "APP_MODE=live but GROQ_API_KEY is not set. Add it to your .env file."
                )
            self.primary_client = openai.OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)

            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                # Use real OpenAI for the second opinion + embeddings
                self.secondary_client = openai.OpenAI(api_key=openai_key)
                self.secondary_model = "gpt-4o-mini"
                self.secondary_is_openai = True
            else:
                # No OpenAI key: still satisfy "2 LLM integrations" by
                # cross-checking against a second, different model family
                # on Groq (smaller/faster Llama variant).
                self.secondary_client = openai.OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
                self.secondary_model = GROQ_SECONDARY_MODEL
                self.secondary_is_openai = False

    # ---------- Primary model: reasoning + tool-calling agent ----------

    def claude_analyze(self, system_prompt: str, user_prompt: str,
                        tools: list[dict] | None = None) -> dict:
        """Kept the name `claude_analyze` for drop-in compatibility with
        the rest of the pipeline; it now calls the primary Groq model."""
        if self.mode != "live":
            return self._mock_primary_response(user_prompt)

        kwargs = dict(
            model=GROQ_PRIMARY_MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.primary_client.chat.completions.create(**kwargs)
        return self._parse_openai_style_response(response)

    def _parse_openai_style_response(self, response) -> dict:
        message = response.choices[0].message
        text = message.content or ""
        tool_calls = []
        for call in (message.tool_calls or []):
            import json
            try:
                args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"name": call.function.name, "input": args, "id": call.id})
        return {"text": text, "tool_calls": tool_calls, "raw": response}

    def _mock_primary_response(self, user_prompt: str) -> dict:
        return {
            "text": (
                "[MOCK PRIMARY MODEL RESPONSE (Groq/Llama 3.3 70B) - set "
                "APP_MODE=live and provide GROQ_API_KEY for a real analysis]\n\n"
                "Overall fit: Moderate-to-strong match. Your experience "
                "aligns with the core responsibilities, but the posting "
                "emphasizes skills that are underrepresented on your resume. "
                "Strengthen the top third of your resume with quantified "
                "achievements that mirror the job's key terms."
            ),
            "tool_calls": [],
            "raw": None,
        }

    # ---------- Secondary model: cross-check ----------

    def openai_analyze(self, system_prompt: str, user_prompt: str) -> dict:
        """Kept the name `openai_analyze` for drop-in compatibility; calls
        real OpenAI if OPENAI_API_KEY is set, otherwise a second Groq model."""
        if self.mode != "live":
            return self._mock_secondary_response(user_prompt)

        response = self.secondary_client.chat.completions.create(
            model=self.secondary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
        )
        return {"text": response.choices[0].message.content, "raw": response}

    def _mock_secondary_response(self, user_prompt: str) -> dict:
        return {
            "text": (
                "[MOCK SECONDARY MODEL RESPONSE - set APP_MODE=live and "
                "provide GROQ_API_KEY (and optionally OPENAI_API_KEY) for "
                "a real cross-check analysis]\n\n"
                "Cross-check: Similar conclusion to the primary model. "
                "Biggest gap appears to be in tooling/technology keywords "
                "rather than core competency; consider a skills section "
                "that explicitly lists the technologies from the posting "
                "that you have used."
            ),
            "raw": None,
        }

    def get_openai_client_for_embeddings(self):
        """Embeddings need a real OpenAI client (Groq does not currently
        serve an embeddings endpoint). Returns None if no OPENAI_API_KEY
        is set, in which case the RAG engine falls back to its
        deterministic mock embedder automatically."""
        return self.secondary_client if self.secondary_is_openai else None
