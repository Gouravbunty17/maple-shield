.PHONY: setup dev demo smoke-demo test test-py test-ui test-compliance build-ui clean

PY := python3
PIP := pip

setup:
	$(PIP) install --break-system-packages -r requirements.txt
	cd operator-ui && npm install

dev:
	@echo "Run each service in its own terminal, or use scripts/dev.sh"
	@bash scripts/dev.sh

demo:
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/demo_live.ps1 $(DEMO_ARGS); \
	elif command -v powershell.exe >/dev/null 2>&1; then \
		powershell.exe -ExecutionPolicy Bypass -File scripts/demo_live.ps1 $(DEMO_ARGS); \
	else \
		bash scripts/demo_live.sh $(DEMO_ARGS); \
	fi

smoke-demo:
	$(PY) scripts/smoke_demo.py $(SMOKE_ARGS)

test: test-compliance test-py build-ui
	@echo "[OK] all checks passed"

test-py:
	$(PY) -m pytest -q

test-ui:
	cd operator-ui && npm run build

test-compliance:
	$(PY) -m pytest -q tests/test_compliance.py

build-ui:
	cd operator-ui && npm run build

clean:
	rm -rf */node_modules */dist */build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
