"""src — the Legal-RAG pipeline.

Each module is a stage, named for what it does, so that a failing stage is
easy to name:

    fetch.py     pinned statutory snapshot from the IndiaCode API  -> data/raw/
    build.py     snapshot + manifest -> one Markdown file per act -> data/corpus/
    ingest.py    Markdown -> chunks + embeddings                   -> data/index/
    retrieve.py  a question -> ranked provisions, measured not assumed
    answer.py    cited answers, with the refusal contract enforced
    evaluate.py  the golden set -> metrics, saved to reports/
    diagnose.py  which of the seven stages failed
    service.py   FastAPI: /health /ask /compare /trace /diagnose /eval
    gate.py      the regression gate: does this change beat the last commit?
    ui.py        Streamlit, for looking at what the retriever actually did

`python -m src.<stage>` runs any of them.
"""