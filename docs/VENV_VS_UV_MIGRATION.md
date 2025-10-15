# Python Environment: .venv vs uv Migration Guide

## Current Setup

### What You Have Now

```bash
$ ls -lha .venv/
drwxr-xr-x  6 satmuri6 gtperson 4.0K Sep 22 16:16 .
-rwxrwxrwx  1 satmuri6 gtperson    0 Sep 22 16:16 .lock
drwxr-xr-x  2 satmuri6 gtperson 4.0K Sep 29 17:12 bin
drwxr-xr-x  3 satmuri6 gtperson 4.0K Sep 22 15:57 include
drwxr-xr-x  3 satmuri6 gtperson 4.0K Sep 22 15:57 lib
lrwxrwxrwx  1 satmuri6 gtperson    3 Sep 22 15:57 lib64 -> lib
-rw-r--r--  1 satmuri6 gtperson  516 Sep 22 15:57 pyvenv.cfg
drwxr-xr-x  3 satmuri6 gtperson 4.0K Sep 22 15:59 share
```

**Current Environment:**
- Standard Python virtual environment (`.venv/`)
- Created with traditional `python -m venv .venv`
- Packages defined in `pyproject.toml`
- Activated with `source .venv/bin/activate`

**Dependencies** (from `pyproject.toml`):
- PyTorch 2.6.0, transformers, accelerate (ML stack)
- DEAP 1.4.1 (genetic algorithms)
- FastAPI, uvicorn (LLM server)
- matplotlib, seaborn, pandas (visualization)
- 20+ total dependencies

**Status:** ✅ Working, but infrastructure team recommends migration

---

## Why Migrate to uv?

[`uv`](https://github.com/astral-sh/uv) is a modern Python package and project manager written in Rust, designed as a drop-in replacement for `pip`, `pip-tools`, `pipx`, and `virtualenv`.

### Key Benefits

1. **Speed** 🚀
   - 10-100x faster than pip for package installation
   - Written in Rust with parallel downloads
   - Intelligent caching reduces redundant downloads

2. **Reproducibility** 🔒
   - Deterministic resolution with `uv.lock` file
   - Faster, more reliable than `pip-compile`
   - Cross-platform consistency

3. **Simplicity** ✨
   - Single tool replaces pip, pip-tools, virtualenv, pipx
   - Drop-in replacement: minimal learning curve
   - Works with existing `pyproject.toml`

4. **Infrastructure Support** 🏢
   - Your team's infrastructure recommends it
   - Likely available cluster-wide soon
   - Industry trend toward Rust-based Python tooling

### Comparison

| Feature | Traditional pip + venv | uv |
|---------|----------------------|-----|
| Install speed | 🐢 Slow | 🚀 10-100x faster |
| Lock file | ❌ Manual with pip-tools | ✅ Built-in `uv.lock` |
| Resolution | ⚠️ Sometimes inconsistent | ✅ Deterministic |
| Tool count | 🔧 Multiple (pip, venv, pip-tools) | 🔧 One tool |
| pyproject.toml | ✅ Supported | ✅ Native support |
| Existing workflow | ✅ No change needed | ⚠️ Migration required |

---

## Migration Path

### Option 1: Gradual Migration (Recommended)

Keep your `.venv` working while testing `uv` on the side.

#### Step 1: Install uv

```bash
# Personal installation (no admin required)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or request cluster-wide installation
# (Your infrastructure team may already be planning this)
```

#### Step 2: Create uv.lock (without affecting .venv)

```bash
# Generate lock file from pyproject.toml
uv lock

# This creates uv.lock with pinned versions
# Your .venv is unaffected
```

#### Step 3: Test in Parallel

```bash
# Create a test environment with uv
uv venv .venv-uv

# Install dependencies
uv sync

# Activate and test
source .venv-uv/bin/activate
python run_improved.py --help  # Test that everything works

# Deactivate when done
deactivate
```

#### Step 4: Switch When Ready

```bash
# Backup old venv
mv .venv .venv-backup

# Create new venv with uv
uv venv .venv

# Install all dependencies
uv sync

# Activate as usual
source .venv/bin/activate

# Test thoroughly before deleting backup
```

### Option 2: Direct Migration

If you're confident and want to switch immediately:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Remove old venv
rm -rf .venv

# Create uv.lock and install
uv sync

# Activate
source .venv/bin/activate
```

---

## Updated Workflow

### With Traditional pip + venv

```bash
# Activate environment
source .venv/bin/activate

# Install new package
pip install some-package

# Update pyproject.toml manually
# Generate requirements.txt with pip freeze (if needed)

# Deactivate
deactivate
```

### With uv

```bash
# Create/update environment (no separate activation needed for most commands)
uv sync

# Add new package
uv add some-package  # Automatically updates pyproject.toml and uv.lock

# Run scripts directly
uv run python run_improved.py

# Or activate as usual
source .venv/bin/activate
python run_improved.py
deactivate
```

---

## Key Commands

### uv Equivalents

| Task | pip + venv | uv |
|------|-----------|-----|
| Create environment | `python -m venv .venv` | `uv venv` |
| Activate | `source .venv/bin/activate` | Same |
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install pkg` | `uv add pkg` |
| Remove package | `pip uninstall pkg` | `uv remove pkg` |
| Update packages | `pip install -U pkg` | `uv lock --upgrade` |
| Run script | `python script.py` | `uv run python script.py` |

### uv-Specific Features

```bash
# Create environment and sync in one step
uv sync

# Run without activating
uv run python run_improved.py

# Add dev dependencies
uv add --dev pytest black

# Update specific package
uv add --upgrade torch

# Lock without installing
uv lock

# Export to requirements.txt (for backwards compatibility)
uv export --format requirements-txt > requirements.txt
```

---

## SLURM Integration

Your SLURM scripts will work the same way:

### Current (with .venv)

```bash
#!/bin/bash
#SBATCH ...

# Activate venv
source /home/hice1/satmuri6/scratch/llm-inference/.venv/bin/activate

# Run script
python run_improved.py
```

### With uv

```bash
#!/bin/bash
#SBATCH ...

# Option 1: Activate venv as before
source /home/hice1/satmuri6/scratch/llm-inference/.venv/bin/activate
python run_improved.py

# Option 2: Use uv run (no activation needed)
cd /home/hice1/satmuri6/scratch/llm-inference
uv run python run_improved.py
```

**No changes needed** to `run.sh` or `server.sh`!

---

## Troubleshooting

### "uv not found"

```bash
# Check if installed
which uv

# If not, install with:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (usually done automatically)
export PATH="$HOME/.cargo/bin:$PATH"
```

### "Package conflict"

```bash
# uv has better resolution than pip
# If you see conflicts, it's catching issues pip missed

# Try forcing resolution
uv lock --resolution highest

# Or pin problematic package
uv add "some-package==1.2.3"
```

### "Slower than expected"

```bash
# First run downloads and builds wheels
# Subsequent runs use cache and are much faster

# Check cache location
uv cache dir

# Clear cache if needed
uv cache clean
```

---

## Recommendation

### For Your Project

1. **Short term:** Keep using `.venv` (it's working!)
2. **Next step:** Test `uv` in parallel (`.venv-uv`)
3. **Migration:** Switch once infrastructure team confirms cluster-wide support

### Timeline Suggestion

- **Week 1:** Install uv locally, generate `uv.lock`, test basic workflows
- **Week 2:** Run full evolution with uv environment, compare performance
- **Week 3:** Switch to uv as default, keep `.venv-backup` for 1-2 weeks
- **Week 4:** Delete backup, commit `uv.lock` to git

---

## Files to Track in Git

### Current

```
.venv/                 # ❌ In .gitignore
pyproject.toml         # ✅ Tracked
```

### After Migration

```
.venv/                 # ❌ Still in .gitignore
.venv-uv/              # ❌ Also ignore
pyproject.toml         # ✅ Tracked
uv.lock                # ✅ Track this! (ensures reproducibility)
```

---

## References

- [uv Documentation](https://github.com/astral-sh/uv)
- [uv vs pip Benchmark](https://github.com/astral-sh/uv#benchmarks)
- [Python Packaging Guide](https://packaging.python.org/)

---

## Questions?

- **Q: Will this break my current setup?**  
  A: No! Migration is optional and gradual. Your `.venv` keeps working.

- **Q: Do SLURM scripts need changes?**  
  A: No! Activation works the same. `uv run` is optional.

- **Q: What if I want to go back?**  
  A: Just activate your `.venv-backup` and you're back to pip.

- **Q: When should I migrate?**  
  A: When infrastructure team confirms cluster support, or when you want better performance.
