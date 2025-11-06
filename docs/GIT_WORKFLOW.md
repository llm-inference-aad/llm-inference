# Git Workflow for LLM-Inference Project

## Branch Strategy

### Main Branches
- **`new_main`** - Production branch (stable, tested code)
- **`develop`** - Integration branch for features (optional)

### Feature Branches
Create feature branches for new work:

```bash
# Create and switch to feature branch
git checkout -b feature/description-of-feature

# Or for bug fixes
git checkout -b fix/description-of-bug
```

### Branch Naming Conventions
- `feature/` - New features (e.g., `feature/distributed-mode`)
- `fix/` - Bug fixes (e.g., `fix/syntax-error-llm-utils`)
- `refactor/` - Code refactoring (e.g., `refactor/improve-fitness-inheritance`)
- `docs/` - Documentation updates (e.g., `docs/add-baseline-guide`)
- `test/` - Testing improvements (e.g., `test/add-unit-tests`)

## Workflow Steps

### 1. Start New Feature
```bash
# Ensure you're on new_main and up to date
git checkout new_main
git pull origin new_main

# Create feature branch
git checkout -b feature/my-new-feature
```

### 2. Make Changes
```bash
# Make your code changes
# ...

# Check status
git status

# Stage changes
git add <files>

# Commit with descriptive message
git commit -m "feat: add new feature X

- Detailed description of what changed
- Why it changed
- Any breaking changes"
```

### 3. Test Before Merging
```bash
# Run syntax check
python -m py_compile src/*.py

# Run a quick test (optional)
# python -m pytest tests/
```

### 4. Merge to Main
```bash
# Switch to main
git checkout new_main

# Merge feature branch (--no-ff preserves branch history)
git merge --no-ff feature/my-new-feature -m "Merge feature/my-new-feature: Brief description"

# Delete feature branch (optional)
git branch -d feature/my-new-feature
```

## Commit Message Format

Use conventional commit format:

```
<type>: <subject>

<body>

<footer>
```

### Types
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `chore:` - Build/config changes

### Examples
```bash
# Good commit messages
git commit -m "feat: add auto seed network training

- Train seed network before evolution runs
- Only runs on fresh starts, not checkpoint resume
- Saves results to network_results.txt for fitness inheritance"

git commit -m "fix: correct indentation in llm_utils fallback logic

- Fix SyntaxError in else block
- Move fail-fast exception into proper control flow"

# Bad commit messages (avoid these)
git commit -m "fixed stuff"
git commit -m "update"
git commit -m "changes"
```

## Pre-Commit Hook

A pre-commit hook is installed to catch syntax errors:

- ✅ Automatically checks Python syntax before allowing commit
- ❌ Blocks commits with syntax errors
- Located at: `.git/hooks/pre-commit`

## Handling Conflicts

If you encounter merge conflicts:

```bash
# See which files have conflicts
git status

# Edit conflicted files (look for <<<<<<< markers)
# ...

# After resolving conflicts
git add <resolved-files>
git commit -m "Merge: resolve conflicts in <files>"
```

## Best Practices

### DO ✅
- Create feature branches for all changes
- Write descriptive commit messages
- Test before merging to main
- Use `--no-ff` for merges to preserve history
- Delete feature branches after merging

### DON'T ❌
- Commit directly to `new_main` (except for trivial changes)
- Use vague commit messages
- Commit code with syntax errors (pre-commit hook prevents this)
- Force push (`git push -f`) unless absolutely necessary
- Leave stale feature branches around

## Common Commands

```bash
# View branch history
git log --oneline --graph --decorate --all

# See what changed
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard uncommitted changes
git checkout -- <file>

# List all branches
git branch -a

# Delete local branch
git branch -d feature/old-feature

# Push to remote
git push origin new_main
```

## Emergency: Reverting Bad Commits

If you committed something broken to main:

```bash
# Revert the last commit (creates new commit)
git revert HEAD

# Or revert specific commit
git revert <commit-hash>

# Or reset to previous state (use carefully!)
git reset --hard <good-commit-hash>
```

## Example Workflow

```bash
# Starting work on new baseline optimization
git checkout new_main
git checkout -b feature/optimize-baseline-metrics

# Make changes to constants.py
vim src/cfg/constants.py

# Commit
git add src/cfg/constants.py
git commit -m "feat: optimize baseline metrics collection

- Add metrics tracking for accuracy and params
- Store metrics in run-specific directories
- Update constants for better baseline reproducibility"

# More changes to run_improved.py
vim run_improved.py
git add run_improved.py
git commit -m "refactor: improve checkpoint loading logic

- Skip seed training when loading checkpoint
- Add clear logging for checkpoint vs fresh start
- Validate checkpoint data before use"

# Ready to merge
git checkout new_main
git merge --no-ff feature/optimize-baseline-metrics -m "Merge feature/optimize-baseline-metrics"

# Clean up
git branch -d feature/optimize-baseline-metrics
```

## Summary

**Golden Rule:** Always work on feature branches, test thoroughly, then merge to `new_main`.

This keeps `new_main` stable and allows you to experiment safely on feature branches.
