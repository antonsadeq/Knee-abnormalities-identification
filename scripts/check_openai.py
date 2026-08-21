"""Verify the OpenAI setup before committing to a 4,407-report run.

    python scripts/check_openai.py                 # key, models, one live call
    python scripts/check_openai.py --model gpt-5

Checks, in the order they fail:

1. the SDK is installed,
2. a key is visible to *this* process (a key set in another terminal is not),
3. the key authenticates and the requested model id exists on the account,
4. the model honours the strict JSON schema the labeller depends on.

Step 4 is the one worth running. Not every model supports ``response_format`` with a strict schema,
and discovering that 500 requests into a full run is an expensive way to learn it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _diagnose(exc: Exception) -> str:
    """Turn an API exception into the one action that actually fixes it.

    Worth doing carefully: these failures arrive at the same call site and look alike, but a quota
    error and a schema rejection need completely different responses. Reporting "this model may not
    support structured output" for what is really an empty account sends you to change the model,
    which cannot possibly help.
    """
    text = str(exc)
    code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)

    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return (
            "   BILLING, not the model or the schema. Your key authenticated and the model exists\n"
            "   (step 3 passed), so this is an empty platform account.\n\n"
            "   An OpenAI API account is funded separately from a ChatGPT Plus/Pro subscription —\n"
            "   the subscription buys the web app and contributes nothing to API quota.\n\n"
            "   Add a payment method or buy credits:\n"
            "     https://platform.openai.com/settings/organization/billing\n"
            "   Then check the balance is non-zero and re-run this script."
        )
    if code == 429 or "rate limit" in text.lower():
        return (
            "   Rate limited (but the account has quota). Wait a minute and retry;\n"
            "   for the full run lower --concurrency."
        )
    if code == 401 or "invalid_api_key" in text or "Incorrect API key" in text:
        return (
            "   The key was rejected. It may have been revoked, or it belongs to a different\n"
            "   organisation than the one you expect. Issue a fresh key at\n"
            "     https://platform.openai.com/api-keys"
        )
    if "response_format" in text or "json_schema" in text or "not supported" in text:
        return (
            "   This model does not support strict structured output. Try another model id;\n"
            "   the labeller depends on schema-constrained JSON."
        )
    if code == 404 or "does not exist" in text:
        return "   Model not found for this account — check the id and your org's access."
    return f"   Unrecognised failure:\n   {text}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    args = p.parse_args()

    print("1. SDK ...", end=" ")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("MISSING.  pip install openai")
    print("ok")

    print("2. API key ...", end=" ")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        sys.exit(
            "NOT SET in this process.\n"
            '   PowerShell (this session):  $env:OPENAI_API_KEY = "sk-..."\n'
            '   Persistent:                 setx OPENAI_API_KEY "sk-..."  (then open a new terminal)'
        )
    if not key.startswith("sk-"):
        print(f"WARNING: does not look like an API key (starts {key[:6]!r})")
    print(f"ok ({key[:7]}...{key[-4:]})")

    client = OpenAI()

    print("3. account + model ...", end=" ")
    try:
        available = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"FAILED\n{_diagnose(exc)}")
    if args.model not in available:
        hint = sorted(m for m in available if m.startswith(("gpt", "o")))[:15]
        sys.exit(f"model {args.model!r} not on this account.\n   candidates: {hint}")
    print(f"ok ({args.model} available, {len(available)} models total)")

    print("4. strict JSON schema ...", end=" ")
    from llm_label import _schema  # noqa: PLC0415

    system = (ROOT / "prompts" / "label_reports.md").read_text(encoding="utf-8")
    probe = [{
        "id": "probe-1",
        "report": (
            "Hallazgos: Menisco lateral de morfologia y senal conservada, sin signos de rotura. "
            "Derrame articular moderado. Rotura del cuerno posterior del menisco medial."
        ),
    }]
    try:
        response = client.chat.completions.create(
            model=args.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"reports": probe}, ensure_ascii=False)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "knee_findings", "strict": True, "schema": _schema()},
            },
        )
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"FAILED\n{_diagnose(exc)}")
    print("ok")

    out = json.loads(response.choices[0].message.content)["reports"][0]
    print("\n--- probe result (Spanish report, three findings planted) ---")
    print(f"  language detected: {out['language']}")
    for key in ("lateral_meniscus", "medial_meniscus", "effusion", "fracture"):
        v = out["findings"][key]
        print(f"  {key:<18} {v['state']:<14} sev={v['severity']:<5} {v['evidence'][:60]}")

    expected = {
        "lateral_meniscus": "absent",       # negation follows the structure
        "medial_meniscus": "present",       # explicit rotura
        "effusion": "present",              # derrame moderado
        "fracture": "not_mentioned",        # silence, not a negative
    }
    wrong = {k: (out["findings"][k]["state"], v) for k, v in expected.items()
             if out["findings"][k]["state"] != v}
    usage = response.usage
    print(f"\n  tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")
    if wrong:
        print("\n  MISMATCH against the expected reading (got, expected):")
        for k, v in wrong.items():
            print(f"    {k}: {v}")
        print("  Read a few more before the full run — the prompt may need tightening.")
    else:
        print("\n  All four probe findings read correctly. Ready for --limit 40.")


if __name__ == "__main__":
    main()
