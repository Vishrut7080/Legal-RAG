"""Report what this machine can actually do, before a pipeline stage fails.

Every other module here raises when something is missing. That is right for a
pipeline and wrong for the first thing you run: `make gate` failing on a missing
sentence-transformers install tells you nothing about why. This module reports
availability instead of raising, and prints one line per item.

Run it with `make check`. It is safe to run offline and safe to run in a broken
environment -- nothing in here can raise on purpose.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (label, import name, human consequence of it being absent)
REQUIRED = [
    ("numpy", "numpy", "retrieval scoring"),
    ("yaml", "pyyaml", "thresholds.yml parsing"),
    ("rank_bm25", "rank_bm25", "the lexical baseline retriever"),
    ("sentence_transformers", "sentence-transformers", "embedding and reranking"),
]

OPTIONAL = [
    ("torch", "torch", "faster CPU inference"),
    ("fastapi", "fastapi", "the HTTP service"),
    ("uvicorn", "uvicorn", "serving the HTTP service"),
    ("streamlit", "streamlit", "the browser UI"),
    ("litellm", "litellm", "any chat model, including local Ollama"),
    ("pandas", "pandas", "tabular analysis of results.json"),
]


def _module_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown version"


def check_packages() -> list[tuple[str, bool, str]]:
    rows = []
    for module, dist, needed in REQUIRED + OPTIONAL:
        present = importlib.util.find_spec(module) is not None
        note = f"{_module_version(dist)} - {needed}" if present else f"MISSING - {needed}"
        rows.append((module, present, note))
    return rows


def check_artifacts() -> list[tuple[str, bool, str]]:
    """Are the build products there? Distinguishing absent from stale matters."""
    rows: list[tuple[str, bool, str]] = []

    def path(rel: str) -> Path:
        return REPO / rel

    for rel, why in [
        ("data/manifest.json", "declares what gets fetched"),
        ("data/raw/FETCHED.json", "provenance and checksums for the snapshot"),
        ("data/corpus/index.json", "the built corpus inventory"),
        ("data/index/index.json", "the retrieval index"),
        ("data/index/chunks.jsonl", "chunk text and metadata"),
        ("data/eval/golden.jsonl", "the evaluation cases"),
    ]:
        p = path(rel)
        if p.exists():
            rows.append((rel, True, f"present ({p.stat().st_size:,} bytes) - {why}"))
        else:
            rows.append((rel, False, f"ABSENT - {why}"))

    raw = path("data/raw/acts")
    n_payloads = len(list(raw.glob("*.json"))) if raw.is_dir() else 0
    rows.append((
        "data/raw/acts/*.json",
        n_payloads > 0,
        f"{n_payloads} raw Act payloads" if n_payloads else "ABSENT - run `make fetch`",
    ))
    return rows


def check_models() -> list[tuple[str, bool, str]]:
    """Whether the models this project will actually load are usable offline.

    Resolution goes through `aip.config.resolve_model` and strips the `local/`
    prefix, because that is what `aip.embed` does. Checking the raw profile
    string instead gives a false negative that looks like a broken install:
    sentence-transformers 6.x reads a leading `local/` as a filesystem path.
    """
    from aip.config import resolve_model

    rows: list[tuple[str, bool, str]] = []
    if importlib.util.find_spec("sentence_transformers") is None:
        rows.append((
            "embedding", False,
            "cannot check - sentence-transformers is not installed",
        ))
        return rows

    from sentence_transformers import SentenceTransformer

    for tier in ("EMBED", "RERANK"):
        model = resolve_model(tier).removeprefix("local/")
        try:
            loaded = SentenceTransformer(model)
            # Renamed in sentence-transformers 6.x; fall back so this probe does
            # not become a version gate.
            dim_of = getattr(loaded, "get_embedding_dimension", None) or (
                loaded.get_sentence_embedding_dimension
            )
            rows.append((f"{tier.lower()}: {model}", True, f"cached, {dim_of()}-dim"))
        except Exception as exc:
            rows.append((
                f"{tier.lower()}: {model}", False,
                f"not usable: {type(exc).__name__}. Run `make index` once with "
                "network access to cache it.",
            ))
    return rows


def check_ollama() -> tuple[bool, str]:
    """Reachable local chat model, or why not. Never raises on a refused port."""
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    host = url.split("//")[-1].split(":")[0]
    port = int(url.rsplit(":", 1)[-1]) if ":" in url.rsplit("//")[-1] else 11434
    if shutil.which("ollama") is None:
        return False, "not installed (only needed for `answer.py --llm`)"
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True, f"reachable at {url}"
    except OSError:
        return False, f"installed but not serving at {url} - run `ollama serve`"


def check_environment() -> list[tuple[str, bool, str]]:
    env = REPO / ".env"
    rows = [
        (".env", env.exists(), "present" if env.exists() else "ABSENT - run `make env`"),
    ]
    if env.exists():
        profile = next(
            (ln.split("=", 1)[1].strip()
             for ln in env.read_text(encoding="utf-8").splitlines()
             if ln.startswith("AIP_PROFILE=")),
            "unset (defaults to local)",
        )
        rows.append(("AIP_PROFILE", True, profile))
    strict = [v for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
              if os.environ.get(v) == "1"]
    rows.append((
        "strict offline mode",
        bool(strict),
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" if len(strict) == 2
        else "not set - Hugging Face will phone home on every run and warn about it",
    ))
    return rows


def main() -> int:
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"repo        {REPO}\n")

    sections = [
        ("packages", check_packages()),
        ("build artifacts", check_artifacts()),
        ("models", check_models()),
        ("environment", check_environment()),
    ]
    for title, rows in sections:
        print(title)
        width = max((len(label) for label, _, _ in rows), default=0)
        for label, present, note in rows:
            print(f"  [{'ok  ' if present else 'WARN'}] {label:<{width}}  {note}")
        print()

    reachable, note = check_ollama()
    print("optional chat model")
    print(f"  [{'ok  ' if reachable else 'WARN'}] {'ollama':<6}  {note}\n")

    hard = all(present for _, present, _ in check_artifacts())
    print("ready to run `make gate`" if hard
          else "run `make fetch`, `make corpus`, `make index` first")
    return 0


if __name__ == "__main__":
    sys.exit(main())