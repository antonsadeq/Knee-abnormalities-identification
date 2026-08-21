"""Push and poll the Kaggle cache kernel, one shard per invocation.

    python scripts/run_cache_kernel.py --split test  --shards 1                  # validation run
    python scripts/run_cache_kernel.py --split train --shards 4 --shard 0
    python scripts/run_cache_kernel.py --split train --shards 4 --shard 0 --fetch artifacts/

The kernel's parameters are literals at the top of ``build_cache.py`` because a Kaggle script takes
no arguments. Rather than editing that file for every run and leaving the repo holding whatever the
last run happened to need, this stages a patched copy in a temp directory and pushes that, so the
committed defaults stay put.

Polling matters more than it looks. ``kernels push`` returns as soon as the version is queued, so
without a wait the next step reads the *previous* run's output — which looks like a successful run
that produced nothing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "notebooks" / "cache_336"
KERNEL_ID = "antonsadeq/rsna-knee-cache-336"
KAGGLE = shutil.which("kaggle") or "kaggle"

#: Kernel worker states that mean the run is over. Matched against the parsed enum name, never as a
#: substring of the command's whole output — a `requests.ConnectionError` traceback contains the word
#: "error", and reading that as *the kernel* having failed is how a transient network drop silently
#: abandons a running job.
TERMINAL = {"COMPLETE", "ERROR", "CANCELACKNOWLEDGED", "CANCELREQUESTED"}

#: Consecutive status checks allowed to fail before giving up. The kernel keeps running on Kaggle
#: regardless of whether we can reach the API, so a blip is worth waiting out rather than abandoning.
MAX_STATUS_FAILURES = 20

STATUS_RE = re.compile(r"KernelWorkerStatus\.([A-Z_]+)")


def patch(source: str, **values) -> str:
    """Rewrite the top-level literal assignments the kernel reads its configuration from.

    Done line by line rather than with a regex so the trailing comments — which carry the reasoning
    for each value — keep their alignment instead of being jammed against the new literal.
    """
    lines = source.splitlines(keepends=True)
    pending = dict(values)

    for i, line in enumerate(lines):
        for key in list(pending):
            if not line.startswith(f"{key} = "):
                continue
            value = pending.pop(key)
            literal = f'"{value}"' if isinstance(value, str) else str(value)
            body, hash_, comment = line.rstrip("\n").partition("#")
            replacement = f"{key} = {literal}"
            if hash_:
                gap = max(len(body) - len(replacement), 1)
                replacement += " " * gap + hash_ + comment
            lines[i] = replacement + "\n"
            break

    if pending:
        raise SystemExit(f"no top-level assignment found for {sorted(pending)}")
    return "".join(lines)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)


def poll_status() -> tuple[str | None, str]:
    """``(state, raw)`` where state is the parsed enum name, or None if the API was unreachable."""
    result = run([KAGGLE, "kernels", "status", KERNEL_ID])
    raw = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return None, raw.splitlines()[-1] if raw else "status call failed"
    found = STATUS_RE.search(raw)
    return (found.group(1).upper() if found else None), raw


def push_with_retry(stage: Path, attempts: int = 5) -> None:
    """Push, retrying transient failures. A push that never lands loses the whole shard."""
    for attempt in range(1, attempts + 1):
        pushed = run([KAGGLE, "kernels", "push", "-p", str(stage)])
        text = (pushed.stdout + pushed.stderr).strip()
        if pushed.returncode == 0:
            print(text)
            return
        print(f"  push attempt {attempt}/{attempts} failed: {text.splitlines()[-1][:200]}")
        if attempt < attempts:
            time.sleep(30 * attempt)
    raise SystemExit("push failed after retries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--fetch", type=Path, default=None, help="download output here when complete")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status checks")
    ap.add_argument("--timeout", type=int, default=8 * 3600)
    ap.add_argument("--no-wait", action="store_true", help="push and return immediately")
    ap.add_argument("--attach", action="store_true",
                    help="do not push; wait on the version already running and fetch it. For "
                         "recovering after a dropped connection, where the kernel kept going on "
                         "Kaggle but the local driver stopped watching.")
    args = ap.parse_args()

    if not 0 <= args.shard < args.shards:
        raise SystemExit(f"--shard must be in [0, {args.shards})")

    if args.attach:
        print(f"attaching to the running version of {KERNEL_ID} (no push)")
    else:
        push(args)

    wait_and_fetch(args)


def push(args) -> None:
    source = (KERNEL_DIR / "build_cache.py").read_text()
    staged = patch(
        source,
        SHARD=args.shard,
        N_SHARDS=args.shards,
        SPLIT=args.split,
        SMOKE_ONLY=args.smoke_only,
        N_SLICES=args.slices,
        SIZE=args.size,
    )

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        (stage / "build_cache.py").write_text(staged)
        shutil.copy2(KERNEL_DIR / "kernel-metadata.json", stage / "kernel-metadata.json")

        print(f"pushing {KERNEL_ID}  split={args.split} shard={args.shard}/{args.shards} "
              f"size={args.size} slices={args.slices} smoke_only={args.smoke_only}")
        push_with_retry(stage)


def wait_and_fetch(args) -> None:
    if args.no_wait:
        print("pushed; not waiting")
        return

    deadline = time.time() + args.timeout
    state, last, failures = None, "", 0
    while time.time() < deadline:
        time.sleep(args.poll)
        state, text = poll_status()

        if state is None:
            failures += 1
            print(f"  [{time.strftime('%H:%M:%S')}] status unavailable "
                  f"({failures}/{MAX_STATUS_FAILURES}): {text[:160]}", flush=True)
            if failures >= MAX_STATUS_FAILURES:
                raise SystemExit("gave up polling; the kernel may still be running on Kaggle")
            continue

        failures = 0
        if text != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {text}", flush=True)
            last = text
        if state in TERMINAL:
            break
    else:
        raise SystemExit("timed out waiting for the kernel")

    if state != "COMPLETE":
        print(f"\nrun ended as {state} — pulling the log anyway")

    if args.fetch:
        args.fetch.mkdir(parents=True, exist_ok=True)
        out = run([KAGGLE, "kernels", "output", KERNEL_ID, "-p", str(args.fetch)])
        print((out.stdout + out.stderr).strip()[:2000])

    print("\ndone")


if __name__ == "__main__":
    main()
