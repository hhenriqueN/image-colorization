"""Generate Kaggle-ready versions of the training notebooks.

Reads notebooks/0{1,2,3}_*.ipynb and emits notebooks/0{1,2,3}_*_kaggle.ipynb
with a Kaggle bootstrap cell prepended and path handling adapted so the same
notebook also runs locally unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"

REPO_URL = "https://github.com/hhenriqueN/image-colorization.git"
DATASET_SLUG = "image-colorization-processed"

BOOTSTRAP_SOURCE = f"""\
# === Kaggle bootstrap ===
# Clones the repo, installs missing deps, and links the uploaded dataset.
# Skips automatically when running locally.

from pathlib import Path
import os, sys, subprocess

IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    REPO_URL  = "{REPO_URL}"
    REPO_DIR  = Path("/kaggle/working/image-colorization")
    WORK_DIR  = Path("/kaggle/working")

    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)

    os.chdir(REPO_DIR)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    subprocess.run(["pip", "install", "-q", "scikit-image"], check=True)

    # Find the uploaded processed dataset (locate `manifest.csv`)
    candidates = list(Path("/kaggle/input").rglob("manifest.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find manifest.csv under /kaggle/input. "
            "Did you attach the '{DATASET_SLUG}' dataset to this notebook?"
        )
    DATASET_DIR = candidates[0].parent
    target = REPO_DIR / "data" / "processed"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink() if target.is_symlink() else None
    if not target.exists():
        target.symlink_to(DATASET_DIR)

    print("Kaggle bootstrap OK")
    print(f"  Repo:    {{REPO_DIR}}")
    print(f"  Dataset: {{DATASET_DIR}}")
else:
    print("Local environment — skipping Kaggle bootstrap.")
"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_cell_source(cell: dict, new_src: str) -> None:
    cell["source"] = new_src.splitlines(keepends=True)


def patch_project_root(src: str) -> str:
    """Replace the original PROJECT_ROOT block with Kaggle-aware logic."""
    new_block = (
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "if Path(\"/kaggle/input\").exists():\n"
        "    PROJECT_ROOT = Path(\"/kaggle/working/image-colorization\")\n"
        "else:\n"
        "    _here = Path.cwd()\n"
        "    PROJECT_ROOT = _here.parent if _here.name == \"notebooks\" else _here\n"
        "\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n"
    )
    return new_block


def patch_cfg(src: str) -> str:
    """Rewrite the CFG cell so checkpoint paths land in /kaggle/working/ on Kaggle."""
    out_lines = []
    for line in src.splitlines():
        if '"checkpoint_dir":' in line:
            # Extract the relative tail after PROJECT_ROOT so we can route the
            # checkpoint dir into /kaggle/working when running on Kaggle.
            indent = line[: len(line) - len(line.lstrip())]
            stripped = line.strip().rstrip(",")
            # stripped looks like:  "checkpoint_dir": PROJECT_ROOT / "checkpoints" / "unet"
            rhs = stripped.split(":", 1)[1].strip()
            out_lines.append(
                f'{indent}"checkpoint_dir": (Path("/kaggle/working") / "checkpoints" / {rhs.split("/")[-1].strip()}) if Path("/kaggle/input").exists() else {rhs},'
            )
        elif "PHASE2_CHECKPOINT" in line and "=" in line and "PROJECT_ROOT" in line:
            # cGAN notebook references prior phase's best.pth
            out_lines.append(
                'PHASE2_CHECKPOINT = (\n'
                '    Path("/kaggle/input/phase2-resnet-unet/best.pth")\n'
                '    if Path("/kaggle/input/phase2-resnet-unet/best.pth").exists()\n'
                '    else (Path("/kaggle/working/checkpoints/resnet_unet/best.pth")\n'
                '          if Path("/kaggle/input").exists()\n'
                '          else PROJECT_ROOT / "checkpoints" / "resnet_unet" / "best.pth")\n'
                ')'
            )
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def transform_notebook(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text())

    cells = nb["cells"]

    # 1. Find and patch the PROJECT_ROOT cell (first code cell that mentions PROJECT_ROOT)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if "PROJECT_ROOT" in src and ("Path(\"__file__\")" in src or "sys.path.insert" in src):
            set_cell_source(cell, patch_project_root(src))
            break

    # 2. Find and patch the CFG cell
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if "checkpoint_dir" in src or ("PHASE2_CHECKPOINT" in src and "best.pth" in src):
            set_cell_source(cell, patch_cfg(src))

    # 3. Insert bootstrap cells after the leading title markdown cell (index 0)
    intro_md = md_cell(
        "## 0. Kaggle bootstrap\n"
        "\n"
        "First-time setup: clone the repo, install missing deps, and link the "
        "uploaded `image-colorization-processed` dataset. **Skips automatically "
        "when running locally**, so the same notebook works in both places.\n"
    )
    bootstrap = code_cell(BOOTSTRAP_SOURCE)

    cells.insert(1, intro_md)
    cells.insert(2, bootstrap)

    # Force the standard python3 kernelspec — Kaggle does not have custom local kernels.
    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.11",
    }

    return nb


def main() -> None:
    pairs = [
        ("01_unet.ipynb", "01_unet_kaggle.ipynb"),
        ("02_resnet_unet.ipynb", "02_resnet_unet_kaggle.ipynb"),
        ("03_cgan.ipynb", "03_cgan_kaggle.ipynb"),
    ]

    for src_name, dst_name in pairs:
        src = NOTEBOOKS_DIR / src_name
        dst = NOTEBOOKS_DIR / dst_name
        nb = transform_notebook(src)
        dst.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"  wrote {dst.relative_to(NOTEBOOKS_DIR.parent)}")


if __name__ == "__main__":
    main()
