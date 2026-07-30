"""
guardrails.py
--------------
Safety and content-moderation layer that sits between raw user input
and the LLM, and between the LLM and the UI.

Three checks, run in order:
  1. PII scrubbing        - mask emails/phones/SSNs before they hit logs
  2. Prompt-injection scan - flag resumes/job posts that try to hijack
                             the assistant's instructions
  3. Content moderation    - block sexual/violent/hateful content using
                             the OpenAI moderation endpoint (falls back
                             to a local keyword check in mock mode)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,2}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

# Phrases that indicate an attempt to override the system prompt / persona
# from *inside* a document the model is asked to analyze (indirect
# prompt injection). Kept as patterns, not an exhaustive list.
INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the) (previous|above|prior) instructions", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"disregard (your|the) (guidelines|rules|instructions)", re.I),
    re.compile(r"reveal (your|the) (system prompt|instructions)", re.I),
]

BLOCK_KEYWORDS = [
    # coarse local fallback used only when APP_MODE=mock and no
    # moderation API is available
    "kill myself", "child sexual", "how to make a bomb",
]


@dataclass
class GuardrailResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    sanitized_text: str = ""


def scrub_pii(text: str) -> str:
    """Mask PII before it is written to any log or eval artifact."""
    scrubbed = text
    scrubbed = PII_PATTERNS["ssn"].sub("[REDACTED-SSN]", scrubbed)
    scrubbed = PII_PATTERNS["email"].sub("[REDACTED-EMAIL]", scrubbed)
    scrubbed = PII_PATTERNS["phone"].sub("[REDACTED-PHONE]", scrubbed)
    return scrubbed


def detect_prompt_injection(text: str) -> list[str]:
    hits = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def local_keyword_check(text: str) -> list[str]:
    lowered = text.lower()
    return [kw for kw in BLOCK_KEYWORDS if kw in lowered]


def moderate_with_openai(text: str, openai_client) -> list[str]:
    """Use OpenAI's moderation endpoint when a live client is available."""
    try:
        resp = openai_client.moderations.create(
            model="omni-moderation-latest", input=text
        )
        result = resp.results[0]
        if result.flagged:
            return [cat for cat, val in result.categories.model_dump().items() if val]
        return []
    except Exception as exc:  # network/key issues shouldn't crash the app
        return [f"moderation_check_failed: {exc}"]


def run_input_guardrails(text: str, openai_client=None, mode: str = "mock") -> GuardrailResult:
    """
    Full guardrail pass on user-supplied text (resume or job description)
    before it is embedded, chunked, or sent to an LLM.
    """
    reasons = []

    injection_hits = detect_prompt_injection(text)
    if injection_hits:
        reasons.append(
            "Possible prompt-injection content detected in the uploaded document; "
            "it will be treated as inert data, not instructions."
        )

    if mode == "live" and openai_client is not None:
        flagged = moderate_with_openai(text, openai_client)
    else:
        flagged = local_keyword_check(text)

    if flagged:
        reasons.append(f"Content moderation flagged: {', '.join(flagged)}")

    sanitized = scrub_pii(text)

    # We block outright only on real moderation flags, not on injection
    # attempts (those are neutralized by prompt design instead, see
    # src/pipeline.py where document content is wrapped and never
    # concatenated directly into the system role).
    allowed = not any("Content moderation flagged" in r for r in reasons)

    return GuardrailResult(allowed=allowed, reasons=reasons, sanitized_text=sanitized)


def run_output_guardrails(text: str, openai_client=None, mode: str = "mock") -> GuardrailResult:
    """Lighter pass on the LLM's own output before showing it to the user."""
    if mode == "live" and openai_client is not None:
        flagged = moderate_with_openai(text, openai_client)
    else:
        flagged = local_keyword_check(text)

    allowed = not flagged
    reasons = [f"Output flagged: {', '.join(flagged)}"] if flagged else []
    return GuardrailResult(allowed=allowed, reasons=reasons, sanitized_text=text)
