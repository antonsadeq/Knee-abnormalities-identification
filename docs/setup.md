# Setup

## Environment

```powershell
conda env create -f environment.yml     # note: "env create", not "create"
conda activate rsna-knee
pip install -e .
```

`conda env create -f` reads a YAML environment file and takes the environment name from its `name:`
field. **`conda create --file` is a different command** — it expects a flat spec list
(`numpy=1.26`, one per line) and fails on YAML with
`CondaValueError: could not parse 'name: rsna-knee'`. To use a different name, add `-n <name>`.

PyTorch is not pinned in `environment.yml` because the CUDA build must match the local driver.
Install it explicitly, then verify:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected here: `Quadro RTX 5000 with Max-Q Design`, 16 GB VRAM.

## Kaggle API

Configured and working. Kaggle CLI ≥ 2.2 uses a bare access token, **not** the legacy
`kaggle.json` — that file is ignored and its presence is misleading. The token lives at
`%USERPROFILE%\.kaggle\access_token`.

To rotate it: generate a new one at <https://www.kaggle.com/settings/api>, then

```powershell
Set-Content "$env:USERPROFILE\.kaggle\access_token" -Value "<token>" -NoNewline -Encoding ascii
kaggle config view          # should report auth_method: ACCESS_TOKEN
```

Verify access:

```powershell
kaggle competitions files rsna-knee-abnormality-detection
```

## Local model (report labelling) — no account, no cost

This is the recommended path. It needs no API key and no card, and it is the better option for the
competition anyway: the labelling is reproducible, the weights and script ship with the code if you
win, and it avoids the question of whether a paid external service counts as "reasonably available
to all participants". Published work finds open-weight models
[competitive with GPT-4o at extracting findings from free-text radiology reports](https://pubs.rsna.org/doi/10.1148/radiol.240895).

### Install

```powershell
winget install --id Ollama.Ollama
```

The installer lands in `%LOCALAPPDATA%\Programs\Ollama`. Start the server and pull a model:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama serve            # leave running, or it starts on demand
ollama pull qwen3:14b   # ~9 GB at 4-bit; fits the 16 GB RTX 5000 comfortably
```

Pick for **multilingual** strength, not benchmark averages — the corpus is nine languages across
three scripts, and Greek, Bulgarian, Croatian and Turkish are where a weak model will quietly fail.
Qwen3-class and Gemma-3-class 12–14B models are the candidates. An 8B runs roughly twice as fast if
throughput turns out to be the binding constraint.

### Run

Same script as the hosted path — only the endpoint changes. Batch size, concurrency and schema mode
adapt automatically when `--base-url` is set.

```powershell
python scripts/llm_label.py --base-url http://localhost:11434/v1 --model qwen3:14b --gold-only
python scripts/compare_labellers.py

python scripts/llm_label.py --base-url http://localhost:11434/v1 --model qwen3:14b
```

Run `--gold-only` first regardless. It is 58 studies, finishes in minutes, and is the only subset
where this model and the rule labeller can both be scored against truth — so it says whether the
full run is worth the GPU hours before you spend them.

**Expect the full corpus to take hours, not minutes.** Output is the binding cost: roughly 540
tokens per report across the twelve findings including the quoted evidence, so ~2.4 M tokens total.
At a typical 25–40 tokens/s for a 14B model on this GPU that is on the order of a day. The script
prints a live rate and ETA from the first few batches — read it, and drop to an 8B model or leave it
running overnight accordingly. The run is resumable, so stopping and restarting costs only the
batches in flight.

Local models honour JSON schemas less reliably than hosted ones, so responses are validated
client-side: a row with a hallucinated study id, a missing finding, or an invalid state is discarded
rather than written, and a later resume retries it. A half-filled row would otherwise leave a
permanent hole in the label matrix.

## OpenAI (report labelling) — the paid alternative

The LLM is used **once, offline**, to derive training targets from the reports. It is not part of
inference — the submission kernel runs with internet disabled and never sees text. So this is a
one-time cost, and the output (`artifacts/llm_labels.jsonl`) is an artifact to keep, not to
regenerate.

### 1. API access is not the ChatGPT subscription

They are separate products, billed separately. A ChatGPT Plus/Pro plan gives you the web app; it
does **not** give you API access. You need a key from
<https://platform.openai.com/api-keys> and a payment method or credit on the *platform* account at
<https://platform.openai.com/settings/organization/billing>. Without credit, every call returns a
quota error regardless of your ChatGPT plan.

### 2. Install and set the key

```powershell
conda activate rsna-knee
pip install openai
$env:OPENAI_API_KEY = "sk-..."          # this terminal only
```

`setx OPENAI_API_KEY "sk-..."` makes it persistent but writes the secret to the user registry and
does **not** affect the session you run it in — which is the usual reason a key "isn't working".
Prefer a git-ignored `.env`, or set it per session. To undo a `setx`:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $null, "User")
```

A key that has been pasted anywhere shared should be revoked at
<https://platform.openai.com/api-keys> and reissued. PowerShell also saves every command you type,
including the key, at `(Get-PSReadlineOption).HistorySavePath`.

### 3. Verify before spending anything

```powershell
python scripts/check_openai.py --model gpt-5
```

Checks the SDK, that the key is visible to *this* process, that the model id exists on the account,
and that it honours the strict JSON schema the labeller depends on. It then labels one planted
Spanish report and compares four findings against the correct reading — negation that follows the
structure, an explicit tear, a graded effusion, and a silence that must not become a negative.
Discovering a schema incompatibility 500 requests into a full run is an expensive way to learn it.

Set `--model` to whatever you actually have; the id is not guessed for you.

### 4. Run

```powershell
python scripts/llm_label.py --model gpt-5 --limit 40    # ~1 minute, read the output
python scripts/llm_label.py --model gpt-5               # full corpus, resumable
python scripts/compare_labellers.py                     # audit vs the rule labeller
```

The run appends to `artifacts/llm_labels.jsonl` and skips whatever is already there, so a rate
limit, a dropped connection or Ctrl-C costs only the batches in flight. Re-run the same command to
continue.

### Capping what can be charged

The platform account bills against **prepaid credits**, and auto-recharge is opt-in. Buy a small
fixed amount, leave auto-recharge **off**, and set a hard monthly usage limit under
*Settings → Limits*. With auto-recharge off, an exhausted balance returns the same
`insufficient_quota` error as an empty account — the job stops, nothing further is charged. Confirm
both settings in the billing UI before the first run.

### Size of the job

Measured against the real corpus (4,407 reports, 1.47 M tokens of report text, a 3.2 k-token system
prompt resent per batch):

| job | reports | requests | input | output |
|---|---:|---:|---:|---:|
| `--gold-only` | 58 | 8 | 0.05 M | 0.03 M |
| full corpus, `--batch-size 16` | 4,407 | 276 | 2.35 M | 2.38 M |
| full corpus, **default batch 8** | 4,407 | 551 | 3.22 M | 2.38 M |
| full corpus, `--batch-size 4` | 4,407 | 1,102 | 4.98 M | 2.38 M |

Cost is those token counts times your account's rates — check
<https://platform.openai.com/docs/pricing>, since it varies by more than an order of magnitude
across models. For orientation, the full corpus lands near **$2 on a small model and $30–50 on a
frontier one**, while `--gold-only` is **cents**.

**Reasoning models bill hidden reasoning tokens as output**, and those can exceed the visible
completion several times over. Check `usage.completion_tokens` on the `--limit 40` run and scale it
up before committing to the full corpus.

Salvaging only the gaps does *not* help: 4,377 of 4,407 reports have at least one silent or hedged
cell, so a salvage pass costs essentially the same as labelling everything.

Tuning: raise `--concurrency` (default 4) if your rate limit allows, and lower `--batch-size` if
responses come back truncated.

## Data

Metadata CSVs (~9 MB total) — already downloaded to `data/`:

```powershell
foreach ($f in 'train.csv','train_series.csv','test.csv','test_series.csv','sample_submission.csv') {
  kaggle competitions download -c rsna-knee-abnormality-detection -f $f -p data
}
```

Listing the full file set is impractical — the API paginates through ~819k DICOM entries — so
download imaging by explicit path rather than by browsing:
`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`.

The DICOM archive is 247 GB zipped (~570 GB extracted). On the current Linux box `/` has 668 GB
free, so the zip plus a cache fits but a full extraction does not. Stage it and point
`RSNA_DATA_ROOT` at it:

```powershell
$env:RSNA_DATA_ROOT = "S:\rsna-knee\data"
$env:RSNA_CACHE_ROOT = "S:\rsna-knee\cache"
```

`data/` and `artifacts/` in this repo are git-ignored; nothing from the competition dataset should
ever be committed.
