"""src — the Legal-RAG pipeline.

Each module is a stage, named for what it does, so that a failing stage is
easy to name:

    fetch.py        pinned statutory snapshot from the IndiaCode API  -> data/raw/
    build.py        snapshot + manifest -> one Markdown file per act -> data/corpus/
    ingest.py       Markdown -> chunks + embeddings                   -> data/index/
    retrieve.py     a question -> ranked provisions, measured not assumed
    answer.py       cited answers, with the refusal contract enforced
    evaluate.py     the golden set -> metrics, saved to data/eval/
    diagnose.py     which of the seven stages failed
    telemetry.py    per-stage latency, per-request cost, /metrics aggregates
    cache.py        exact + semantic answer cache, with the scope guard
    cache_sweep.py  measures the semantic threshold instead of asserting it
    service.py      FastAPI: /health /ask /ask/stream /retrieve /provision /metrics
    gate.py         the regression gate: does this change beat the last commit?
    break_it.py     breaks the pipeline on purpose, to prove the gate bites
    ui.py           Streamlit, for looking at what the retriever actually did

`python -m src.<stage>` runs any of them.
"""