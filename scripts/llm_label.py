"""Label the report corpus with an LLM, as an independent second opinion to the rule labeller.

Works against any OpenAI-compatible endpoint. Local is the recommended path — no account, no cost,
and reproducible enough to ship with the code:

    ollama serve && ollama pull qwen3:14b
    python scripts/llm_label.py --base-url http://localhost:11434/v1 --model qwen3:14b --gold-only
    python scripts/compare_labellers.py

Hosted, if you would rather pay for it:

    $env:OPENAI_API_KEY = "sk-..."
    python scripts/llm_label.py --gold-only          # the cheap decisive experiment
    python scripts/llm_label.py                      # full corpus

Design notes:

* **Resumable.** Results append to a JSONL checkpoint keyed by study id, and a re-run skips whatever
  is already there. Neither a 4,407-report API run nor a day-long local one survives its first
  interruption otherwise, and losing completed work to a retry is avoidable.
* **Batched.** Several reports per request amortises the long system prompt. Batches stay small so
  the model attends to each report properly and one malformed response costs little — smaller still
  for local models, which drift on long outputs.
* **Schema-enforced, then verified anyway.** Hosted models are constrained by a JSON schema built
  from the label list; local runtimes honour that less reliably, so every row is validated
  client-side and a bad one is dropped rather than written. A half-filled row would leave a
  permanent hole in the label matrix, whereas a dropped one is retried on the next resume.
* **Deterministic-ish.** Temperature 0 and a fixed batch order. Re-running still will not reproduce
  bit-for-bit; the labels are therefore an artifact to keep, not to regenerate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, REPORT, STUDY_ID  # noqa: E402

PROMPT_PATH = ROOT / "prompts" / "label_reports.md"

#: JSON-safe keys for the twelve findings, in the order of ``LABELS``. The competition's own column
#: names contain spaces and an apostrophe, which are awkward in a schema and invite quoting bugs.
KEYS = [
    "acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
    "effusion", "synovitis", "bakers", "contusion", "fracture",
]
KEY_TO_LABEL = dict(zip(KEYS, LABELS))

STATES = ["present", "absent", "uncertain", "not_mentioned"]


#: Appended to the system prompt in compact mode. Evidence quotes are roughly two thirds of the
#: output tokens, and on a throughput-bound local run that is the difference between a job that
#: finishes overnight and one that does not finish at all.
NO_EVIDENCE_OVERRIDE = """

---

# OVERRIDE: omit evidence

This run does not collect evidence quotes. Ignore the "Evidence" section above and the `evidence`
field in the example. Emit only `state` and `severity` for each finding.

Everything else is unchanged. In particular you must still read the negation scope before deciding
a state — you are simply not quoting the span you used.
"""


def _schema(evidence: bool = True) -> dict:
    props: dict[str, dict] = {
        "state": {"type": "string", "enum": STATES},
        "severity": {"type": "number"},
    }
    if evidence:
        props["evidence"] = {"type": "string"}
    finding = {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "language": {"type": "string"},
                        "findings": {
                            "type": "object",
                            "properties": {k: finding for k in KEYS},
                            "required": KEYS,
                            "additionalProperties": False,
                        },
                    },
                    "required": ["id", "language", "findings"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["reports"],
        "additionalProperties": False,
    }


def _validate(row: object, sent: set[str], evidence: bool = True) -> dict | None:
    """Return a well-formed report row, or None if it cannot be trusted.

    Local open-weight models honour a JSON schema less reliably than the hosted ones, so the result
    is checked rather than assumed. A row that fails is simply not written, which means a later
    resume will retry it — silently accepting a half-filled row would put a permanent hole in the
    label matrix instead.
    """
    if not isinstance(row, dict):
        return None
    sid = row.get("id")
    # A hallucinated id would attach one report's findings to a different study.
    if sid not in sent:
        return None

    findings = row.get("findings")
    if not isinstance(findings, dict):
        return None

    clean: dict[str, dict] = {}
    for key in KEYS:
        verdict = findings.get(key)
        if not isinstance(verdict, dict) or verdict.get("state") not in STATES:
            return None
        try:
            severity = float(verdict.get("severity", 0.0))
        except (TypeError, ValueError):
            severity = 0.0
        clean[key] = {"state": verdict["state"], "severity": min(max(severity, 0.0), 1.0)}
        if evidence:
            clean[key]["evidence"] = str(verdict.get("evidence", ""))[:400]
    return {"id": sid, "language": str(row.get("language", "unk")), "findings": clean}


class Fatal(RuntimeError):
    """A failure that retrying cannot fix, so the whole run should stop."""


#: Substrings identifying errors that will fail identically on every batch. Retrying them wastes
#: five backoff sleeps per batch and then does it again for every remaining batch — 29 batches of a
#: missing model is 145 pointless requests before the run reports anything useful.
_PERMANENT = (
    "not found",
    "does not exist",
    "invalid_api_key",
    "incorrect api key",
    "insufficient_quota",
    "model_not_found",
)


def ollama_transport(base_url: str, model: str, evidence: bool, timeout: float = 1800.0):
    """Talk to Ollama's own endpoint rather than its OpenAI-compatible one.

    This exists for one measured reason. Qwen3-class models reason by default, and Ollama strips the
    trace out of ``content`` while still generating and charging for it — a request whose whole
    answer is ``{"ok":true}`` came back at **288 tokens** through ``/v1`` and **16** through
    ``/api/chat`` with ``think=false``. On a throughput-bound GPU that ratio is the entire
    difference between a job that finishes and one that does not.

    Neither ``/no_think`` in the prompt (which made it *worse*, 420 tokens) nor
    ``chat_template_kwargs.enable_thinking`` had any effect through the OpenAI-compatible path.
    ``/api/chat`` also takes a JSON schema in ``format``, so nothing is given up by using it.
    """
    import httpx

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = f"{root}/api/chat"

    def send(system: str, payload: str) -> str:
        response = httpx.post(
            url,
            json={
                "model": model,
                "stream": False,
                "think": False,
                "format": _schema(evidence),
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": payload},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"] or ""

    return send


def openai_transport(client, model: str, schema_mode: str, evidence: bool):
    formats = {
        "strict": {
            "type": "json_schema",
            "json_schema": {"name": "knee_findings", "strict": True, "schema": _schema(evidence)},
        },
        "json_object": {"type": "json_object"},
        "none": None,
    }

    def send(system: str, payload: str) -> str:
        kwargs = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": payload},
            ],
        }
        if formats[schema_mode] is not None:
            kwargs["response_format"] = formats[schema_mode]
        return client.chat.completions.create(**kwargs).choices[0].message.content or ""

    return send


def _call(
    transport,
    system: str,
    batch: list[dict],
    evidence: bool = True,
    attempts: int = 5,
) -> list[dict]:
    payload = json.dumps({"reports": batch}, ensure_ascii=False)
    sent = {b["id"] for b in batch}

    for attempt in range(attempts):
        try:
            content = transport(system, payload)
            # Some local runtimes wrap JSON in prose or a fenced block.
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                content = content[start : end + 1]

            rows = json.loads(content).get("reports", [])
            valid = [r for r in (_validate(r, sent, evidence) for r in rows) if r]
            if valid:
                return valid
            raise ValueError(f"no valid rows in a response of {len(rows)}")
        except Exception as exc:  # noqa: BLE001 — any transient failure is worth one more try
            text = str(exc).lower()
            if any(p in text for p in _PERMANENT):
                raise Fatal(str(exc)) from exc
            if attempt == attempts - 1:
                print(f"  ! batch failed after {attempts} attempts: {exc}", flush=True)
                return []
            # Jittered backoff: a rate limit hit by every worker at once must not resynchronise.
            time.sleep(min(60, 2**attempt) * (0.5 + random.random()))
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"),
                   help="whichever model you have access to; verify the id before a full run")
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible endpoint. For a local Ollama server pass "
        "http://localhost:11434/v1 — no API key or account needed. Defaults for batch size, "
        "concurrency and schema mode adapt automatically when this is set.",
    )
    p.add_argument("--batch-size", type=int, default=0, help="0 = pick by backend")
    p.add_argument("--concurrency", type=int, default=0, help="0 = pick by backend")
    p.add_argument(
        "--schema-mode",
        choices=["strict", "json_object", "none"],
        default=None,
        help="how hard to constrain the output. Hosted models honour 'strict'; local runtimes vary, "
        "so they default to 'json_object' with client-side validation.",
    )
    p.add_argument("--limit", type=int, default=0, help="label only the first N reports")
    p.add_argument(
        "--gold-only",
        action="store_true",
        help="label only the 58 studies that carry image-derived labels. This is the experiment "
        "that decides whether the full run is worth paying for: it is the only subset where the "
        "LLM and the rule labeller can both be scored against truth, and it costs under a dollar.",
    )
    p.add_argument(
        "--no-evidence",
        action="store_true",
        help="skip the quoted evidence span. Cuts output tokens by roughly two thirds, which on a "
        "throughput-bound local GPU is the difference between finishing overnight and not "
        "finishing. Keep evidence for --gold-only and for whatever you intend to adjudicate.",
    )
    p.add_argument("--out", default=str(ARTIFACT_ROOT / "llm_labels.jsonl"))
    args = p.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")

    local = bool(args.base_url)
    # A local model generates far more slowly and drifts on long outputs, so it gets small batches
    # and no concurrency — one GPU cannot serve four requests in parallel anyway.
    if not args.batch_size:
        args.batch_size = 2 if local else 8
    if not args.concurrency:
        args.concurrency = 1 if local else 4
    if args.schema_mode is None:
        args.schema_mode = "json_object" if local else "strict"

    if local:
        # Ollama and friends ignore the key but the SDK insists on one being present. The SDK client
        # is still built, for the model-availability preflight below.
        client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"))
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("set OPENAI_API_KEY")
        client = OpenAI()

    # Preflight: confirm the model exists before queuing hundreds of batches against it. Ollama in
    # particular leaves a half-pulled model absent from the registry while its blobs sit on disk,
    # which looks like a completed download until the first request 404s.
    try:
        available = {m.id for m in client.models.list()}
        if args.model not in available:
            hint = ", ".join(sorted(available)[:10]) or "(none)"
            sys.exit(
                f"model {args.model!r} is not available at "
                f"{args.base_url or 'the OpenAI API'}.\n"
                f"  available: {hint}\n"
                + ("  pull it with:  ollama pull " + args.model if local else "")
            )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — listing is a convenience, not a requirement
        print(f"  (could not list models: {exc}; continuing)")

    evidence = not args.no_evidence
    transport = (
        ollama_transport(args.base_url, args.model, evidence)
        if local
        else openai_transport(client, args.model, args.schema_mode, evidence)
    )

    system = PROMPT_PATH.read_text(encoding="utf-8")
    if args.no_evidence:
        system += NO_EVIDENCE_OVERRIDE
    train = pd.read_csv(DATA_ROOT / "train.csv")
    if args.gold_only:
        train = train[train[LABELS].notna().all(axis=1)]
        print(f"gold subset: {len(train)} annotated studies")
        # A separate checkpoint, so the cheap experiment cannot be mistaken for a full corpus run
        # later, and so a subsequent full run does not skip these studies.
        if args.out == str(ARTIFACT_ROOT / "llm_labels.jsonl"):
            args.out = str(ARTIFACT_ROOT / "llm_labels_gold.jsonl")
    if args.limit:
        train = train.head(args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001 — a truncated final line is expected after a kill
                    pass
        print(f"resuming: {len(done)} reports already labelled")

    todo = [
        {"id": r[STUDY_ID], "report": str(r[REPORT])}
        for _, r in train.iterrows()
        if r[STUDY_ID] not in done and isinstance(r[REPORT], str)
    ]
    if not todo:
        print("nothing to do")
        return

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    where = args.base_url if local else "the OpenAI API"
    print(
        f"{len(todo)} reports in {len(batches)} batches on {args.model} via {where}"
        f"  [batch={args.batch_size} concurrency={args.concurrency}"
        f"{' think=off schema=native' if local else f' schema={args.schema_mode}'}]"
    )

    lock = threading.Lock()
    state = {"n": 0, "rows": 0, "t0": time.time(), "fatal": ""}

    def work(batch: list[dict]) -> None:
        if state["fatal"]:
            return
        try:
            results = _call(transport, system, batch, evidence)
        except Fatal as exc:
            with lock:
                if not state["fatal"]:
                    state["fatal"] = str(exc)
                    print(f"\nFATAL, stopping: {exc}", flush=True)
            return
        with lock, out.open("a", encoding="utf-8") as fh:
            for row in results:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            state["n"] += 1
            state["rows"] += len(results)
            every = 10 if not local else 5
            if state["n"] % every == 0 or state["n"] == len(batches):
                # A local run is measured in hours, so report the projection rather than a bare
                # count — it is the number that decides whether to switch to a smaller model.
                elapsed = time.time() - state["t0"]
                rate = state["rows"] / elapsed if elapsed else 0
                left = (len(todo) - state["rows"]) / rate if rate else 0
                print(
                    f"  {state['n']}/{len(batches)} batches, {state['rows']} reports, "
                    f"{rate * 60:.1f}/min, ~{left / 3600:.1f} h left",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, batches))

    if state["fatal"]:
        sys.exit(1)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
