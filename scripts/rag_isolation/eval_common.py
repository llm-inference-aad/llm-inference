"""Shared helpers for the RAG isolation eval phase.

Both ``eval_submit.py`` and ``parent_fitness.py`` build the same kind of sbatch
script and copy the same kind of network file into ``sota/ExquisiteNetV2/models``.
We factor that out here to keep the two CLI scripts thin.

The bash template comes from ``src/cfg/constants.PYTHON_BASH_SCRIPT_TEMPLATE``
(unmodified, per spec §3). We just compose ``python_runline`` and submit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SOTA_ROOT = ROOT_DIR / "sota" / "ExquisiteNetV2"
SOTA_MODELS_DIR = SOTA_ROOT / "models"


def _load_template() -> str:
    """Pull the sbatch template from production constants without side effects.

    Importing ``cfg.constants`` triggers torch + RAG initialization, which is
    too heavyweight for a CLI utility. Inline-extract the template instead.
    """
    text = (SRC_DIR / "cfg" / "constants.py").read_text(encoding="utf-8")
    m = re.search(
        r'PYTHON_BASH_SCRIPT_TEMPLATE\s*=\s*"""(.*?)"""',
        text,
        flags=re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not locate PYTHON_BASH_SCRIPT_TEMPLATE in constants.py")
    return m.group(1)


PYTHON_BASH_SCRIPT_TEMPLATE = _load_template()


@dataclass
class EvalJob:
    """Information needed to build and submit one eval sbatch script."""

    eval_gene_id: str   # used as `models.network_{eval_gene_id}` and result file basename
    network_src: Path   # absolute path of the network.py to copy into SOTA_ROOT/models
    run_dir: Path       # where train.py will write {RUN_DIR}/results/{eval_gene_id}_results.txt
    epochs: int = 8
    seed: int = 21
    batch_size: int = 216
    end_lr: float = 0.001
    val_r: float = 0.2
    amp: bool = True
    wall_time: str = "00:30:00"   # tightened from default 8h


def copy_network_into_models(src: Path, eval_gene_id: str) -> Path:
    """Copy ``src`` to ``SOTA_ROOT/models/network_{eval_gene_id}.py``.

    Returns the destination path. Removes any stale ``.fallback`` marker so a
    previous failure for the same gene_id can't poison this one.
    """
    SOTA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = SOTA_MODELS_DIR / f"network_{eval_gene_id}.py"
    shutil.copyfile(src, dst)
    fallback = dst.with_suffix(dst.suffix + ".fallback")
    if fallback.exists():
        fallback.unlink()
    return dst


def remove_network_from_models(eval_gene_id: str) -> None:
    """Delete the SOTA-models copy after fitness has been collected."""
    dst = SOTA_MODELS_DIR / f"network_{eval_gene_id}.py"
    if dst.exists():
        dst.unlink()


def results_path_for(run_dir: Path, eval_gene_id: str) -> Path:
    """Where train.py will write the per-gene results file when ``RUN_DIR`` is set."""
    return run_dir / "results" / f"{eval_gene_id}_results.txt"


def parse_results_file(path: Path) -> dict | None:
    """Parse ``test_acc,total_params,val_acc,tr_time``. Return None if invalid."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 2:
        return None
    try:
        out = {
            "test_acc": float(parts[0]),
            "num_params": float(parts[1]),
        }
    except ValueError:
        return None
    if len(parts) >= 3:
        try:
            out["val_acc"] = float(parts[2])
        except ValueError:
            out["val_acc"] = None
    if len(parts) >= 4:
        try:
            out["tr_time"] = float(parts[3])
        except ValueError:
            out["tr_time"] = None
    return out


def build_python_runline(job: EvalJob) -> str:
    """Compose the ``python train.py ...`` line, mirroring run_improved.submit_run."""
    train_script = SOTA_ROOT / "train.py"
    flags = [
        "-bs", str(job.batch_size),
        "-network", f'"models.network_{job.eval_gene_id}"',
        "-data", "./cifar10",
        "-end_lr", str(job.end_lr),
        "-seed", str(job.seed),
        "-val_r", str(job.val_r),
        "-epoch", str(job.epochs),
    ]
    if job.amp:
        flags.append("-amp")
    return (
        # Inline env exports so train.py picks up RUN_DIR / EPOCHS regardless of
        # how sbatch propagates the parent shell environment.
        f'RUN_DIR={job.run_dir} EPOCHS={job.epochs} '
        f'python {train_script} ' + " ".join(flags)
    )


def build_bash_script(job: EvalJob, slurm_log_dir: Path, slurm_error_dir: Path) -> str:
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    slurm_error_dir.mkdir(parents=True, exist_ok=True)
    base = PYTHON_BASH_SCRIPT_TEMPLATE.format(
        python_runline=build_python_runline(job),
        slurm_log_dir=str(slurm_log_dir),
        slurm_error_dir=str(slurm_error_dir),
        root_dir=str(ROOT_DIR),
    )
    # Tighten wall time for 8-epoch jobs (spec §11). The template's hard-coded
    # ``-t 8:00:00`` is too generous and tends to lengthen the queue.
    base = base.replace("#SBATCH -t 8:00:00", f"#SBATCH -t {job.wall_time}")
    return base


def write_bash_script(job: EvalJob, sh_dir: Path, slurm_log_dir: Path,
                      slurm_error_dir: Path) -> Path:
    sh_dir.mkdir(parents=True, exist_ok=True)
    sh_path = sh_dir / f"{job.eval_gene_id}_eval.sh"
    sh_path.write_text(build_bash_script(job, slurm_log_dir, slurm_error_dir))
    return sh_path


def submit_sbatch(sh_path: Path) -> tuple[bool, str | None, str]:
    """Submit a bash script via ``sbatch``. Returns (ok, job_id, raw_output)."""
    try:
        result = subprocess.run(
            ["sbatch", str(sh_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return False, None, "sbatch not found on PATH"
    except subprocess.TimeoutExpired:
        return False, None, "sbatch timed out"
    raw = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return False, None, raw.strip()
    # Standard sbatch output: "Submitted batch job 12345"
    m = re.search(r"Submitted batch job (\d+)", result.stdout or "")
    job_id = m.group(1) if m else None
    return True, job_id, raw.strip()


def squeue_state(job_id: str) -> str | None:
    """Return the slurm state for a running/pending job, or None if not in squeue."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    state = (result.stdout or "").strip().splitlines()
    return state[0] if state else None


def sacct_final_state(job_id: str) -> str | None:
    """Return the final state of a completed slurm job via sacct."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-o", "State", "-X"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [l.strip() for l in (result.stdout or "").splitlines() if l.strip()]
    return lines[0] if lines else None


def derive_run_stamp(run_dir: Path) -> str:
    """Derive a per-run prefix to avoid collisions in ``models/`` (spec §11)."""
    return run_dir.name


def make_eval_gene_id(run_stamp: str, gene_id: str) -> str:
    """Compose the run-prefixed eval gene_id used in ``models/`` and results paths."""
    return f"{run_stamp}__{gene_id}"
