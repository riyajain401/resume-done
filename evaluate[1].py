"""
evaluate.py
------------
Runs the pipeline against evaluation/test_cases.json and writes a
metrics report to evaluation/EVALUATION_REPORT.md.

Run with:  python -m evaluation.evaluate      (from the project root)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import CareerCoachPipeline
from src import guardrails


def load_case_text(case: dict, key_file: str, key_inline: str) -> str:
    if key_inline in case:
        return case[key_inline]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), case[key_file])
    with open(path, encoding="utf-8") as f:
        return f.read()


def run_evaluation(mode: str = "mock") -> dict:
    cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    passed = 0
    total_latency = 0.0

    for case in cases:
        resume_text = load_case_text(case, "resume_file", "resume_inline")
        job_text = load_case_text(case, "job_file", "job_inline")

        pipeline = CareerCoachPipeline(mode=mode)

        start = time.time()
        result = pipeline.run(resume_text, job_text)
        latency = time.time() - start
        total_latency += latency

        lo, hi = case["expected_score_range"]
        score_ok = lo <= result.match_score <= hi
        blocked_ok = result.blocked == case.get("expect_blocked", False)

        injection_ok = True
        if case.get("expect_injection_flag"):
            injection_ok = any("prompt-injection" in n for n in result.guardrail_notes)

        pii_ok = True
        if case.get("expect_pii_scrubbed"):
            scrubbed = guardrails.scrub_pii(resume_text)
            pii_ok = "[REDACTED-SSN]" in scrubbed and "[REDACTED-EMAIL]" in scrubbed

        case_pass = score_ok and blocked_ok and injection_ok and pii_ok
        passed += int(case_pass)

        rows.append({
            "id": case["id"],
            "description": case["description"],
            "score": result.match_score,
            "expected_range": case["expected_score_range"],
            "score_ok": score_ok,
            "blocked": result.blocked,
            "blocked_ok": blocked_ok,
            "injection_ok": injection_ok,
            "pii_ok": pii_ok,
            "latency_sec": round(latency, 3),
            "pass": case_pass,
        })

    summary = {
        "mode": mode,
        "total_cases": len(cases),
        "passed": passed,
        "pass_rate_pct": round(100 * passed / len(cases), 1),
        "avg_latency_sec": round(total_latency / len(cases), 3),
        "rows": rows,
    }
    return summary


def render_markdown_report(summary: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Mode: `{summary['mode']}`  ",
        f"Test cases: {summary['total_cases']}  ",
        f"Passed: {summary['passed']} / {summary['total_cases']} "
        f"({summary['pass_rate_pct']}%)  ",
        f"Average latency per case: {summary['avg_latency_sec']}s",
        "",
        "## Per-case results",
        "",
        "| ID | Description | Score | Expected range | Blocked | "
        "Injection check | PII check | Latency (s) | Result |",
        "|----|-------------|-------|------------------|---------|"
        "------------------|-----------|--------------|--------|",
    ]
    for r in summary["rows"]:
        result_icon = "PASS" if r["pass"] else "FAIL"
        lines.append(
            f"| {r['id']} | {r['description']} | {r['score']} | "
            f"{r['expected_range']} | {r['blocked']} | "
            f"{'ok' if r['injection_ok'] else 'FAIL'} | "
            f"{'ok' if r['pii_ok'] else 'FAIL'} | {r['latency_sec']} | {result_icon} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `mock` mode measures pipeline correctness (parsing, guardrails, "
        "RAG retrieval, keyword scoring) without incurring API cost or "
        "depending on network access. Re-run with `mode=\"live\"` and real "
        "API keys to also evaluate LLM response quality.",
        "- The deterministic keyword match score is model-independent by "
        "design, so it can be evaluated and regression-tested without an "
        "LLM at all - only the free-text coaching narrative depends on "
        "Claude/GPT.",
        "- TC-05 and TC-06 specifically test the safety guardrail layer: "
        "prompt-injection attempts are detected and neutralized (not "
        "blindly followed), and PII is redacted before being used "
        "downstream.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    mode = os.environ.get("APP_MODE", "mock")
    summary = run_evaluation(mode=mode)
    report = render_markdown_report(summary)

    out_path = os.path.join(os.path.dirname(__file__), "EVALUATION_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport written to {out_path}")
