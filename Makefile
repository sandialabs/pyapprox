# PyApprox development tasks
# Run `make help` to see available targets.

.PHONY: help test test-slow test-slower test-slowest test-all test-minimal \
        lint typecheck docs docs-serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------- Testing ----------

test:  ## Run fast tests (default)
	pytest pyapprox -v --tb=short

test-slow:  ## Run slow tests only (>5s)
	PYAPPROX_RUN_SLOW=1 pytest pyapprox -v --tb=short -m slow

test-slower:  ## Run slower tests only (>30s)
	PYAPPROX_RUN_SLOWER=1 pytest pyapprox -v --tb=short -m slower

test-slowest:  ## Run slowest tests only (>60s)
	PYAPPROX_RUN_SLOWEST=1 pytest pyapprox -v --tb=short -m slowest

test-all:  ## Run all tests including slowest
	PYAPPROX_RUN_SLOWEST=1 pytest pyapprox -v --tb=short

KEEP_VENV ?= 0
MINIMAL_ENV ?= pyapprox-minimal
SOURCE_ENV ?= linalg

test-minimal:  ## Run tests with optional deps removed (KEEP_VENV=1 to keep env)
	@conda env remove -n $(MINIMAL_ENV) --yes --quiet 2>/dev/null || true
	conda create -n $(MINIMAL_ENV) --clone $(SOURCE_ENV) --yes --quiet
	conda run -n $(MINIMAL_ENV) pip uninstall -y scikit-fem cvxpy pyrol umbridge joblib mpire numba 2>/dev/null || true
	conda run -n $(MINIMAL_ENV) pytest pyapprox -v --tb=short -x
	@if [ "$(KEEP_VENV)" = "0" ]; then conda env remove -n $(MINIMAL_ENV) --yes --quiet; \
		echo "Minimal tests passed — env cleaned up"; \
	else echo "Minimal tests passed — env kept as $(MINIMAL_ENV)"; fi

test-minimal-clean:  ## Remove leftover minimal env
	conda env remove -n $(MINIMAL_ENV) --yes --quiet 2>/dev/null || true

# ---------- Linting ----------

lint:  ## Run ruff linter
	ruff check pyapprox/

typecheck:  ## Run mypy type checker
	mypy pyapprox/

# ---------- Documentation ----------

docs:  ## Build tutorial site (parallel)
	cd tutorials && ./build.sh -j auto

docs-serve:  ## Build and serve tutorial site locally
	cd tutorials && ./build.sh -j auto --serve

# ---------- Cleanup ----------

clean:  ## Remove build artifacts
	rm -rf tutorials/library/_site tutorials/library/_build_logs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
