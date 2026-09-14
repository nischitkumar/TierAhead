.PHONY: setup test test-fast doctor baseline sim roofline validate dashboard clean

VENV := .venv
PY := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[predictor,dashboard,parquet,dev]"

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/ -q -k "not mixtral and not Mixtral"

doctor:
	$(PY) -m tierahead.cli doctor

# The flagship end-to-end demo: normal (HBM-only) baseline vs CXL-tiered,
# on the pilot's real committed traces. No GPU needed.
baseline:
	$(PY) -m tierahead.cli baseline --model olmoe --residency-pct 25 --bw-gbps 32
	$(PY) -m tierahead.cli baseline --model mixtral --residency-pct 25 --bw-gbps 32 --precision nf4

sim:
	$(PY) -m tierahead.cli sim --model olmoe --residency-pct 25 --bw-gbps 32

roofline:
	$(PY) -m tierahead.cli roofline --model olmoe
	$(PY) -m tierahead.cli roofline --model mixtral

validate:
	$(PY) -m tierahead.cli validate --model olmoe
	$(PY) -m tierahead.cli validate --model mixtral

dashboard:
	$(PY) -m tierahead.cli dashboard

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
