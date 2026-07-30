# Documented prompts

All prompts live in code (not hidden in strings scattered around), so
they're easy to review and modify. This file explains the intent behind
each one.

## 1. System prompt (`src/pipeline.py::SYSTEM_PROMPT`)

Sent as the `system` role to both Claude and GPT. Key design choices:

- **Explicitly tells the model to treat resume/job content as data, not
  instructions.** This is the primary defense against prompt injection:
  since the system prompt is separate from user content and instructs
  the model to ignore embedded commands, a resume containing "ignore
  previous instructions" cannot actually redirect the model's behavior.
- **Asks for a structured but free-text answer** (fit assessment,
  concrete improvements, missing qualifications) rather than rigid JSON,
  because the output is meant to be read directly by a person, not
  parsed by more code.
- **Tells the model tools are available and when to use them**, so the
  agent decides autonomously rather than the app hard-coding tool calls.

## 2. User prompt template (`src/pipeline.py::CareerCoachPipeline.run`)

Built dynamically per request from:
- an optional role hint
- the top-k RAG-retrieved resume excerpts (not the whole resume — this
  keeps the prompt focused on what's actually relevant to this job)
- the job description (or the text fetched from a URL)
- the deterministic keyword match/miss lists, so the LLM's qualitative
  analysis is grounded in the same facts as the auditable score, rather
  than inventing its own independent judgment of what's "missing"

## 3. Tool descriptions (`src/agent_tools.py::TOOL_SCHEMAS`)

Each tool's `description` field is written as the primary steering
mechanism for *when* Claude should call it — Anthropic's tool-use API
relies on this text for the model's decision, so it's written narrowly
("use this when the user pastes a link instead of text") rather than
vaguely ("gets job info").

## 4. Mock responses (`src/llm_clients.py`)

Not real prompts, but worth documenting: every mock response is
prefixed with `[MOCK ... RESPONSE]` so it's never mistaken for a real
model output during grading or a demo.
