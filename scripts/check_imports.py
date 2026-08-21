"""Confirm every third-party module the code imports is declared in environment.yml.

Trimming a dependency list is how you turn a working environment into one that fails on a machine you
cannot see, so this closes the loop: it walks the actual imports and checks each against the manifest.

    python scripts/check_imports.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Import name -> the distribution that provides it, where the two differ.
PROVIDED_BY = {
    "cv2": "opencv-python-headless",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "httpx": "openai",          # pulled in as a dependency of the openai client
}

#: Installed separately so the CUDA build matches the driver — see docs/setup.md.
OUT_OF_BAND = {"torch", "torchvision", "triton"}


def local_modules() -> set[str]:
    return {p.stem for p in (ROOT / "src" / "rsna_knee").glob("*.py")} | {
        "rsna_knee", "llm_label", "test_dicom", "build_folds",
    }


def imports_in(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.update([node.module.split(".")[0]])
    return found


declared = set()
for line in (ROOT / "environment.yml").read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("- ") and not stripped.startswith("- pip:"):
        name = stripped[2:].split("#")[0].split("=")[0].split(">")[0].strip()
        if name and name not in {"pip", "python"}:
            declared.add(name.lower())

used: dict[str, set[str]] = {}
for path in list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("scripts/*.py")) \
        + list(ROOT.glob("tests/*.py")) + list(ROOT.glob("notebooks/*/*.py")):
    for module in imports_in(path):
        used.setdefault(module, set()).add(path.relative_to(ROOT).as_posix())

stdlib = set(sys.stdlib_module_names)
mine = local_modules()
missing: list[tuple[str, str]] = []

print("=== third-party imports ===")
for module in sorted(used):
    if module in stdlib or module in mine or module.startswith("_"):
        continue
    distribution = PROVIDED_BY.get(module, module).lower()
    if module in OUT_OF_BAND:
        print(f"[ note ] {module:14} installed separately (CUDA-specific)")
    elif distribution in declared:
        print(f"[  ok  ] {module:14} -> {distribution}")
    else:
        where = ", ".join(sorted(used[module])[:2])
        print(f"[ BAD  ] {module:14} not declared — imported by {where}")
        missing.append((module, where))

print("\n=== declared but never imported ===")
names = {PROVIDED_BY.get(m, m).lower() for m in used}
#: Required but never imported directly, so a naive "unused" sweep would delete them. pyarrow is the
#: engine pandas needs for parquet — dropping it is exactly how this project once broke.
INDIRECT = {"pyarrow", "pyyaml", "tqdm", "pytest", "ruff", "psutil", "kaggle", "pip", "conda-forge"}
unused = sorted(declared - names - INDIRECT)
print("  " + (", ".join(unused) if unused else "none"))
print("  exempt (needed but not imported directly): " + ", ".join(sorted(INDIRECT - {"pip"})))

print()
if missing:
    print(f"{len(missing)} undeclared import(s) — add them to environment.yml")
    sys.exit(1)
print("Every third-party import is declared.")
