"""Check the *compute* environment and print the training command tuned for this machine.

`verify_local.py` checks the data; this checks the hardware and the install, and prints the flags that
suit them. Two of these checks exist because of failures this project actually had:

* **Architecture support.** Kaggle's `enable_gpu` hands out an sm_60 P100 that its own PyTorch build
  cannot use, and every CUDA call raises. A card being present is not the same as a card being usable,
  so the compute capability is checked against the arch list the installed torch was built for.
* **torch.compile.** Worth a measured 1.45x on this model (271 -> 187 ms/step) on Linux, but Inductor
  needs Triton and Triton support on Windows is partial. Better to find that out in ten seconds here
  than after a fold has been queued behind it.

    python scripts/preflight.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, BAD = "  ok  ", " warn ", " BAD  "
problems: list[str] = []
notes: list[str] = []
can_compile = False
suggested_fraction = None


def say(status: str, what: str, detail: str = "") -> None:
    print(f"[{status}] {what}" + (f" — {detail}" if detail else ""))
    if status == BAD:
        problems.append(what)


print("=== python and platform ===")
say(OK, "python", f"{sys.version.split()[0]} on {sys.platform}")
windows = sys.platform.startswith("win")
if windows:
    notes.append(
        "Windows: DataLoader workers use spawn, not fork, so each one re-imports the module and "
        "re-pickles the target frames. Start with --workers 4 and raise it only if the GPU is idle; "
        "12 is tuned for Linux fork semantics."
    )

print("\n=== packages ===")
for module in ("torch", "timm", "pandas", "pyarrow", "numpy", "cv2", "pydicom", "sklearn"):
    try:
        loaded = __import__(module)
        say(OK, module, getattr(loaded, "__version__", "?"))
    except ImportError:
        say(BAD, module, "not importable")

print("\n=== gpu ===")
try:
    import torch

    if not torch.cuda.is_available():
        say(BAD, "cuda", "torch.cuda.is_available() is False — training would fall back to CPU, "
                         "which is roughly 40x slower here")
    else:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        arches = {int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")}
        usable = major * 10 + minor in arches
        say(OK if usable else BAD, "device", f"{name}, sm_{major}{minor}")
        if not usable:
            say(BAD, "arch support", f"this torch supports {sorted(arches)} — reinstall a build that "
                                     f"includes sm_{major}{minor}")

        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        free = torch.cuda.mem_get_info()[0] / 2**30
        say(OK, "memory", f"{total:.1f} GiB total, {free:.1f} GiB free now")

        # 16 slices at batch 2 peaked at 7.94 GiB compiled on this project's measurements; without
        # gradient checkpointing the same config extrapolates to ~25 GiB, so keep it on below 26 GiB.
        need = 8.5
        if free < need:
            say(BAD, "headroom", f"{free:.1f} GiB free but ~{need} GiB needed for 16 slices at batch 2")
        else:
            fraction = min(0.90, max(0.20, (free - 1.5) / total))
            suggested_fraction = fraction
            say(OK, "suggested --memory-fraction", f"{fraction:.2f} "
                f"({fraction * total:.1f} GiB, leaving ~1.5 GiB for the display/OS)")
            notes.append(f"Keep gradient checkpointing ON: without it this config needs ~25 GiB, "
                         f"more than this card's {total:.0f} GiB.")

        print("\n=== torch.compile (worth 1.45x if it works) ===")
        try:
            import torch.nn as nn

            probe = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(8, 8, 3, padding=1)).cuda()
            compiled = torch.compile(probe)
            compiled(torch.randn(2, 1, 64, 64, device="cuda")).sum().backward()
            can_compile = True
            say(OK, "torch.compile", "works — pass --compile")
        except Exception as exc:  # noqa: BLE001
            say(WARN, "torch.compile", f"unavailable ({type(exc).__name__}) — omit --compile; "
                                       f"expect ~8 min/epoch instead of ~5.5")
except ImportError:
    say(BAD, "torch", "not installed")

print("\n=== disk and ram ===")
usage = shutil.disk_usage(ROOT)
free_gb = usage.free / 2**30
say(OK if free_gb > 60 else BAD, "free disk", f"{free_gb:.0f} GiB "
    "(need ~45 for the cache, ~3 GiB of checkpoints for 5 folds, plus room to work)")
try:
    import psutil

    ram = psutil.virtual_memory().total / 2**30
    say(OK if ram >= 32 else WARN, "ram", f"{ram:.0f} GiB — an epoch reads 38 GB, so anything under "
        "~48 GiB re-reads from disk each epoch (fine on NVMe, slow on SATA)")
except ImportError:
    print("[      ] ram — install psutil to report this")

print("\n=== notes ===")
for note in notes:
    print(f"  * {note}")

print("\n=== suggested command ===")
compile_flag = " --compile" if can_compile else ""
fraction_text = f"{suggested_fraction:.2f}" if suggested_fraction else "<see above>"
print(f"""  python scripts/train.py --fold 0 --epochs 24 --batch 2 --accum 4 --slices 16 \\
      --drop-path 0.1 --slot-dropout 0.1 --patience 5 --ema 0.999 --tta \\
      --memory-fraction {fraction_text} --workers {4 if windows else 12}{compile_flag} \\
      --tag M4_f0_convnext_tiny_16sl --out artifacts/runs

  Run scripts/verify_local.py first — it exits 1 when the cache is incomplete, and a partial
  cache trains without complaining.""")

print()
if problems:
    print(f"{len(problems)} blocking problem(s): {', '.join(problems)}")
    sys.exit(1)
print("Compute environment is ready.")
