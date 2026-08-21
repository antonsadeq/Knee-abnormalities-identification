# Runbook: finish the report labelling

Every command below is copy-pasteable from the repo root in PowerShell. The whole sequence is
resumable, so it is safe to shut the machine down at any point — including in the middle of the
full-corpus run — and pick up where it stopped.

```powershell
cd c:\personal_work\rsna-knee-abnormality-detection
conda activate rsna-knee
```

## 0. Clean the cooling first

The GPU was measured pinned at **300 MHz** — its idle clock — drawing 27–34 W of 90 W at 78–81 °C
under load, with throttle mask `0x24` (software thermal slowdown + power cap). It idles at 76 °C,
where a healthy card sits near 45. That is roughly a fifth of normal speed, and it turns a 10-hour
job into a 6-day one. Changing the Windows power plan does nothing for it.

So: power down, clean the intake and exhaust vents and the fan blades, and if the machine is old
enough to be out of warranty, repaste. Then verify in step 1 before committing to a long run.

Everything else here works either way — it is only the full-corpus run in step 5 that needs the GPU
healthy.

## 1. Bring Ollama up and check the GPU

```powershell
.\scripts\ollama_up.ps1 -Benchmark
```

This stops any stale servers (two competing for port 11434 is what silently forced CPU inference
earlier), starts exactly one with an 8192-token context, confirms the model is actually registered
rather than half-pulled, prints clocks/power/temperature/throttle reasons, and measures throughput.

Read the projected full-corpus time it prints:

| tok/s | projection | verdict |
|---|---|---|
| ≥ 25 | ≤ 10 h | healthy — proceed |
| 10–25 | 10–25 h | workable overnight |
| < 10 | > 25 h | still throttled, or use a smaller model |

**4.5 tok/s was the throttled measurement.** A healthy Quadro RTX 5000 should reach 25–35.

If it is still slow after cleaning, drop to a smaller model — `ollama pull qwen3:8b` is roughly
twice as fast, `qwen3:4b` about four times, at some cost in multilingual accuracy that step 3 will
measure for you.

## 2. Label the 58 annotated studies

```powershell
python scripts/llm_label.py --base-url http://localhost:11434/v1 --model qwen3:14b --gold-only
```

Writes `artifacts/llm_labels_gold.jsonl`. Evidence quotes are kept here — this is the subset you may
want to read by hand. Add `--no-evidence` if it is running too slowly to bear.

### Reasoning is disabled automatically — do not remove that

Passing an Ollama `--base-url` switches the script to Ollama's **native** `/api/chat` endpoint with
`think: false`, instead of its OpenAI-compatible `/v1`. This is not a stylistic preference. Qwen3
reasons by default, and through `/v1` Ollama strips the trace out of the response while still
generating and charging for every token of it. Measured on this machine with a request whose entire
answer is `{"ok":true}`:

| path | completion tokens |
|---|---:|
| `/v1` | 288 |
| `/v1` with `/no_think` in the prompt | 420 (worse) |
| `/v1` with `chat_template_kwargs.enable_thinking=false` | 288 (no effect) |
| **`/api/chat` with `think:false`** | **16** |

Eighteen times fewer tokens for an identical answer. `/api/chat` accepts a JSON schema in `format`,
so the structured-output guarantee is unchanged.

## 3. Decide whether the full run is worth it

```powershell
python scripts/compare_labellers.py
```

Prints cells asserted per finding, coverage by language, polarity outliers, and gold agreement for
**both** labellers side by side.

What to look for, in order:

1. **Cells asserted.** Half of all rule-labeller cells are silent or hedged. If the LLM asserts on
   substantially more *and* its gold agreement holds up, that is the win — converting silence into
   correct labels matters more than sharpening cells that already carry one.
2. **Gold agreement.** The rule labeller's weak points are Contusion (precision 0.50), Lateral OA
   (0.56) and MCL (0.57). If the LLM does not beat those, it is not worth the GPU hours.
3. **Coverage by language.** Watch Greek, Bulgarian, Turkish and Croatian specifically. A 14B model
   that is fluent in English and weak in Greek will show up here, and swapping models is one
   `ollama pull` away.

If the LLM is level with the rules, stop. The rule labeller is already committed and calibrated, and
a public notebook reached 0.906 on rules alone.

## 4. Run the full corpus

```powershell
python scripts/llm_label.py --base-url http://localhost:11434/v1 --model qwen3:14b --no-evidence
```

`--no-evidence` drops the quoted span, about two thirds of the output tokens. The model is still
instructed to read negation scope; it just does not quote what it used. Keep evidence only for the
gold subset and anything you intend to adjudicate by hand.

Progress prints reports/min and a live ETA. Check it after the first few batches — that number, not
the benchmark, is the real projection.

Interrupt with Ctrl-C whenever you like and re-run the identical command to resume; completed
studies are skipped. Rows that come back malformed are discarded rather than written, so a resume
retries them.

## 5. Merge, audit, adjudicate

```powershell
python scripts/compare_labellers.py
```

Now over the whole corpus. It writes `artifacts/llm_targets.parquet`, `llm_weights.parquet`,
`llm_evidence.parquet` and `artifacts/adjudication_queue.csv`.

The queue is the deliverable. It samples rule-vs-LLM disagreements **stratified by language**, which
is the only scrutiny French (81 reports, zero gold studies), German (262, two) and Dutch (153, two)
can get. Read roughly twenty per language, machine-translating where needed. It is a few hours and
the highest information per hour available on this project.

## 6. Re-calibrate and commit

```powershell
python scripts/calibrate_targets.py
git add artifacts/llm_labels.jsonl
git commit -m "Add LLM report labels"
```

Calibration refits the severity → P(gold) map now that the targets have changed.

Commit `llm_labels.jsonl` deliberately, overriding `.gitignore` with `-f` if needed. It cost hours
of GPU time and it is not bit-for-bit reproducible even at temperature 0 — it is an artifact to
keep, not to regenerate.

## Troubleshooting

| symptom | cause |
|---|---|
| `model 'x' not found` | half-pulled model: blobs on disk, no manifest. Step 1 re-pulls it. |
| Everything is ~10× slow, `ollama ps` shows `100% CPU` | two servers on port 11434. Step 1 fixes it. |
| Labels are nonsense but no errors | context below the 3.2k-token prompt, truncating it silently. Step 1 sets 8192. |
| `pip install openai` from a script that worked before | wrong interpreter — `conda activate rsna-knee` first. |
| Run stalls with no output | first request loads ~9 GB into VRAM; allow a minute. |
