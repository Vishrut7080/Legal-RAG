.PHONY: help venv setup env check fetch sanitise verify corpus index answer diagnose compare evaluate gate gate-demo gate-record cache-sweep bench verify-ci service ui test lint clean

VENV := .venv
PYTHON := $(shell \
  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then echo "$$VIRTUAL_ENV/bin/python"; \
  elif [ -x "$(VENV)/bin/python" ]; then echo "$(VENV)/bin/python"; \
  elif [ -x "$(VENV)/Scripts/python.exe" ]; then echo "$(VENV)/Scripts/python.exe"; \
  elif command -v python >/dev/null 2>&1; then echo python; \
  else echo python3; fi)

BOOTSTRAP := $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)

help:
	@echo "make setup     create .venv and install dependencies (~1.8 GB, mostly PyTorch)"
	@echo "make env       create .env from the template (does not overwrite)"
	@echo "make check     verify the environment and report what is available"
	@echo ""
	@echo "Pipeline, in order. fetch is the only target that needs the network."
	@echo "  make fetch     download the pinned provisions from the IndiaCode API"
	@echo "  make corpus    rebuild data/corpus/ from the committed snapshot"
	@echo "  make index     chunk the corpus, embed it, export data/index/"
	@echo "make verify     check every raw file against FETCHED.json (no network)"
	@echo ""
	@echo "Querying and measuring."
	@echo "  make answer Q=\"what is cheating?\""
	@echo "  make diagnose Q=\"what is cheating?\"    explain the answer, stage by stage"
	@echo "  make compare   score every retriever against the golden set"
	@echo "  make evaluate  end-to-end run -> data/eval/results.json"
	@echo "  make gate      regression gate against thresholds.yml"
	@echo ""
	@echo "Serving."
	@echo "  make service   FastAPI on http://127.0.0.1:8000 (no auth - keep it local)"
	@echo "  make ui        Streamlit on http://127.0.0.1:8501"
	@echo ""
	@echo "make test      run the tests"
	@echo "make lint      ruff"
	@echo "make clean     remove caches and traces"
	@echo ""
	@echo "using: $(PYTHON)"

venv:
	@if [ ! -x "$(VENV)/bin/python" ] && [ ! -x "$(VENV)/Scripts/python.exe" ]; then \
	  echo "creating $(VENV) with $(BOOTSTRAP)..."; \
	  $(BOOTSTRAP) -m venv $(VENV); \
	else echo "$(VENV) already exists"; fi

setup: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Next: make env, then make check"

# `.env.example` starts with a dot, so Finder and Explorer hide it by default.
env:
	@if [ -f .env ]; then \
	  echo ".env already exists -- not touching it."; \
	else \
	  cp .env.example .env; \
	  echo "Created .env from the template."; \
	fi

check:
	$(PYTHON) -m src.check

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

# Network. Downloads every provision named in data/manifest.json and writes the
# raw API responses to data/raw/. Idempotent: existing files are kept unless
# --force, so re-running costs nothing.
fetch:
	$(PYTHON) -m src.fetch

# No network. Normalises data/raw/ into the markdown documents that get indexed.
# This is the reproducibility guarantee: the corpus is a pure function of the
# committed snapshot.
corpus:
	$(PYTHON) -m src.build

# No network, if the embedding model is already cached. Chunks the corpus,
# embeds it, and writes data/index/.
index:
	$(PYTHON) -m src.ingest

# No network. Every raw file against its recorded SHA-256.
verify:
	$(PYTHON) -m src.fetch --check
	$(PYTHON) -m src.validate_golden
	$(PYTHON) -m src.build --stats

# --------------------------------------------------------------------------
# Querying and measuring
# --------------------------------------------------------------------------

# Offline by default. Add --llm to generate with a local chat model instead
# (needs Ollama running); see README "Answering".
answer:
	$(PYTHON) -m src.answer -q "$(Q)"

# The same pipeline, narrated stage by stage: guards -> retrieval -> status
# filter -> sufficiency -> citation. When an answer is wrong, this names the
# stage responsible instead of leaving you to guess.
diagnose:
	$(PYTHON) -m src.diagnose -q "$(Q)"

compare:
	$(PYTHON) -m src.retrieve --compare

evaluate:
	$(PYTHON) -m src.evaluate

gate:
	$(PYTHON) -m src.gate

# Prove the gate can fail. Three break modes: the retrieval window, a corpus
# document, and the index checksum. Exits non-zero if the gate *passes* on any
# of them, so "the gate is green" and "the gate still bites" are separate facts.
gate-demo:
	@for mode in k corpus index; do \
	  echo "=== break mode: $$mode ==="; \
	  $(PYTHON) -m src.break_it --mode $$mode || exit $$?; \
	done

# Regenerate the committed transcript in reports/gate_failure_demo.txt.
gate-record:
	$(PYTHON) -m src.record_gate_demo

# Measure the semantic cache threshold instead of asserting one. Writes
# data/eval/cache_sweep.json, which is the artefact the shipped 0.95 comes from.
cache-sweep:
	$(PYTHON) -m src.cache_sweep --min 0.80 --max 0.995 --step 0.005

# Latency and cost. Two runs, because the two numbers answer different
# questions: with the embedding cache off, every query is a genuinely new
# question; with it on, a repeated question is nearly free. Reporting only the
# second would overstate the service.
bench:
	BENCH_NO_EMBED_CACHE=1 $(PYTHON) -m src.benchmark --n 40
	$(PYTHON) -m src.benchmark --n 40 --out data/eval/benchmark_embed_cache_on.json

# Everything CI runs, in one command, on a clean checkout.
verify-ci: check lint test gate cache-sweep

# Offline. Rewrites data/raw/ under the current storage policy: keeps the
# judgment counts, drops the eCourts judgment records. See LICENSING.md.
sanitise:
	$(PYTHON) -m src.fetch --sanitise

# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

# Loopback only and no authentication. Binding to anything else prints a
# warning; putting this on a network without an authenticating proxy in front
# of it is not something the README can stop you from doing.
service:
	$(PYTHON) -m src.service

ui:
	$(PYTHON) -m streamlit run src/ui.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .aip_traces .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +