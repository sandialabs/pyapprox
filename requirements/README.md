# Requirements

## Hash-Pinned Lockfiles

To generate hash-pinned lockfiles for reproducible installs:

```bash
pip-compile --generate-hashes --output-file requirements/prod.txt pyproject.toml
pip-compile --generate-hashes --extra dev --output-file requirements/dev.txt pyproject.toml
```

These lockfiles can then be installed with:

```bash
pip install --require-hashes -r requirements/prod.txt
```

> **Note**: Hash-pinned lockfiles are not yet generated for this project.
> This directory is a placeholder for future lockfile infrastructure.

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) for local security
checks (gitleaks, bandit, pip-audit, branch protection).

```bash
pip install pre-commit
pre-commit install
```

To run all hooks manually:

```bash
pre-commit run --all-files
```
