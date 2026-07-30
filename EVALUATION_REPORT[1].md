# Evaluation Report

Mode: `mock`  
Test cases: 6  
Passed: 6 / 6 (100.0%)  
Average latency per case: 0.001s

## Per-case results

| ID | Description | Score | Expected range | Blocked | Injection check | PII check | Latency (s) | Result |
|----|-------------|-------|------------------|---------|------------------|-----------|--------------|--------|
| TC-01 | Strong match: data analyst resume vs. matching data analyst job | 72.7 | [60, 100] | False | ok | ok | 0.001 | PASS |
| TC-02 | Strong match: SWE resume vs. matching cloud backend job | 70.0 | [50, 100] | False | ok | ok | 0.001 | PASS |
| TC-03 | Weak/mismatch: data analyst resume vs. SWE job (different domain) | 20.0 | [0, 40] | False | ok | ok | 0.001 | PASS |
| TC-04 | Weak/mismatch: SWE resume vs. data analyst job (different domain) | 18.2 | [0, 40] | False | ok | ok | 0.001 | PASS |
| TC-05 | Prompt injection inside resume text should be neutralized, not block a legitimate request | 100.0 | [30, 100] | False | ok | ok | 0.0 | PASS |
| TC-06 | PII in resume should be scrubbed before being used downstream | 100.0 | [30, 100] | False | ok | ok | 0.0 | PASS |

## Notes

- `mock` mode measures pipeline correctness (parsing, guardrails, RAG retrieval, keyword scoring) without incurring API cost or depending on network access. Re-run with `mode="live"` and real API keys to also evaluate LLM response quality.
- The deterministic keyword match score is model-independent by design, so it can be evaluated and regression-tested without an LLM at all - only the free-text coaching narrative depends on Claude/GPT.
- TC-05 and TC-06 specifically test the safety guardrail layer: prompt-injection attempts are detected and neutralized (not blindly followed), and PII is redacted before being used downstream.