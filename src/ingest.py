"""Chunk the corpus and build a committable offline index.

    python -m src.ingest              # data/corpus/ -> data/index/
    python -m src.ingest --stats      # shape of the current index
    python -m src.ingest --rebuild    # discard and re-embed

Output, all under `data/index/`:

    chunks.jsonl    one line per chunk: chunk_id, doc_id, provision, meta, text
    matrix.npy      float32 [n_chunks, dim], L2-normalised, in chunks.jsonl order
    index.json      embedding model, dims, counts, and the sha256 of chunks.jsonl

Why the index is a commit
-------------------------
At this size the matrix is a couple of megabytes. Committing it means:

  - `git clone` gives you a working retrieval system with no model download and
    no embedding step, so a reviewer can run `src/retrieve.py` immediately.
  - The regression gate in `src/gate.py` compares against a committed
    baseline. If the index is regenerated on every run, every metric drifts
    for reasons that have nothing to do with your change.
  - `AIP_OFFLINE=1` works, which is what makes the gate honest.

This is the right call at ~1k chunks and the wrong call at 1M. The interface
below is deliberately narrow so that swapping in a real vector store later
touches this file only.

Chunking strategy, and why it is not `aip.chunking.markdown_chunks`
---------------------------------------------------------------------
`markdown_chunks` splits on headings and prepends the heading path, which is
right in general and wrong here for one specific reason: it returns metadata
containing only the heading string. This corpus needs `in_force`, `role`,
`superseded_by`, and `replaces` on every chunk, because those are what let a
retriever report that it returned a repealed provision and let the evaluator
score it. So this module does its own heading split to recover that metadata,
and delegates only the oversized-body case to `aip.chunking`.

The resulting chunk text is deliberately not the rendered Markdown block. It
is rebuilt as three lines of status plus the operative text, because the
rendered block carries a source URL, a judgment count, and offence-table rows
that dominate the embedding of a short provision without telling the retriever
anything it can act on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aip.chunking import Chunk, _recursive_split
from aip.config import resolve_model
from aip.embed import embed_batch

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data" / "corpus"
INDEX = REPO / "data" / "index"

CHUNKS = INDEX / "chunks.jsonl"
MATRIX = INDEX / "matrix.npy"
INDEX_JSON = INDEX / "index.json"

# A provision body longer than this is split into several chunks that share one
# `provision` meta value. Roughly 900-1100 tokens of statute prose: long enough
# that a Definitions section keeps its defined terms of art together, short
# enough that a chunk is one answerable question.
CHUNK_CHARS = 2600

_H2 = re.compile(r"^##\s+(.*)$", re.MULTILINE)

# A status line is "- <anything without a colon>: <value>". The key is
# deliberately loose: build.py emits keys like
# "Repealed on 2024-07-01; replaced by BNS s.103 (near-identical, score 0.9)",
# and an alphabetic-only pattern silently fails on those -- which is how a
# repealed provision ends up with its status block glued to the front of the
# body and its superseded_by link missing from the metadata.
_STATUS = re.compile(r"^-\s+([^:]+):\s*(.*)$")


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def split_provisions(markdown: str) -> Iterator[tuple[str, str]]:
    """Yield (heading, block body) for every `##` section of a corpus document."""
    matches = list(_H2.finditer(markdown))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        yield m.group(1).strip(), markdown[m.end() : end].strip()


def parse_status(block: str) -> tuple[dict[str, str], str]:
    """Pull the leading `- Key: value` status lines off the top of a block.

    Returns (status dict, remaining body).

    Consumption stops at the first blank line once at least one status line has
    been read. That is what makes this safe on bodies which are themselves
    bullet lists: `build.py` always writes a blank line between the status
    block and the operative text, so body bullets are never mistaken for
    status.
    """
    lines = block.split("\n")
    status: dict[str, str] = {}
    i = 0
    while i < len(lines) and i < 30:
        line = lines[i].strip()
        if not line:
            if status:
                i += 1
            break
        m = _STATUS.match(line)
        if not m:
            break
        status[_norm_key(m.group(1))] = m.group(2).strip()
        i += 1
    return status, "\n".join(lines[i:]).strip()


def _split_ref(value: str) -> tuple[str, str]:
    """'IPC s.302 (near-identical, score 0.9)' -> ('IPC s.302', 'near-identical')."""
    ref = value.split("(")[0].strip().rstrip(",").strip()
    rel = value.split("(")[1].split(",")[0].strip() if "(" in value else ""
    return ref, rel


def supersession_of(status: dict[str, str]) -> tuple[str, str, str]:
    """(direction, counterpart ref, relation) from a status block.

    `build.py` writes one of:
        - Replaced: IPC s.302 (near-identical, score 0.9)      -> successor
        - Repealed by: BNS s.103 (near-identical, score 0.9)   -> predecessor
    """
    for key in ("replaced_by", "replaced"):
        if (value := status.get(key)):
            if value.lower().startswith("none"):
                continue
            ref, rel = _split_ref(value)
            return ("predecessor" if key == "replaced_by" else "successor"), ref, rel
    return "", "", ""


def prologue(status: dict[str, str]) -> str:
    """The status header prepended to a provision's body in the chunk text.

    Three things are kept deliberately:

      - the citation label and heading, so the text stands alone;
      - the in-force status, because an embedding model cannot see metadata and
        "REPEALED" must be part of the vector if it is going to be filtered on;
      - the supersession link, so a question about the old provision retrieves
        the new one and vice versa.

    Everything else `build.py` rendered -- source URL, judgment count, raw
    offence table -- is metadata, not text. It stays in `meta` where it can be
    reported and scored.
    """
    provision = status.get("provision", "")
    heading = status.get("heading", "")
    title = f"{provision} - {heading}".strip(" -")

    notes: list[str] = []
    if status.get("in_force", "").startswith("yes"):
        notes.append("Current law.")
    else:
        notes.append("REPEALED - historical text only, not in force.")
    direction, ref, rel = supersession_of(status)
    tentative = "tentative" in (rel or "").lower()
    if direction == "predecessor":
        # `build.py` marks a weak correspondence map row as tentative; carrying
        # that caveat into the embedded text matters more here than anywhere,
        # because this string is what the retriever actually matches and quotes.
        # Dropping it re-introduces the flat claim the hedge exists to prevent.
        notes.append(
            f"Replaced by {ref} (tentative map match, confirm against the "
            f"commencement notification)." if tentative else f"Replaced by {ref}."
        )
    elif direction == "successor":
        notes.append(
            f"Replaces {ref} (tentative map match)." if tentative else f"Replaces {ref}."
        )

    return f"{title}\n{' '.join(notes)}".strip()


def load_corpus() -> list[tuple[str, str]]:
    """(doc_id, markdown) for every corpus document, in stable order."""
    index = json.loads((CORPUS / "index.json").read_text(encoding="utf-8"))
    out = []
    for doc in index["docs"]:
        path = REPO / doc["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python -m src.build` first."
            )
        out.append((doc["doc_id"], path.read_text(encoding="utf-8")))
    return out


# --------------------------------------------------------------------------
def chunk_corpus() -> list[Chunk]:
    """Markdown corpus -> provision-level chunks with full metadata."""
    chunks: list[Chunk] = []
    n_oversized = 0

    for doc_id, markdown in load_corpus():
        for heading, block in split_provisions(markdown):
            status, body = parse_status(block)
            if not body:
                continue

            # "CPC O.7 r.11" and "BNS s.103" both come out of - Provision: -
            provision = status.get("provision", heading.split(" - ")[0])
            in_force = status.get("in_force", "").startswith("yes")
            meta: dict[str, Any] = {
                "provision": provision,
                "heading": status.get("heading", ""),
                "act": doc_id,
                "act_short": provision.split()[0] if provision else doc_id,
                "role": "current" if in_force else "repealed",
                "in_force": "True" if in_force else "False",
                "url": status.get("source", ""),
                "strategy": "provision",
            }
            if (j := status.get("reported_judgments_citing_it")):
                meta["n_judgments"] = j

            direction, ref, relation = supersession_of(status)
            if direction == "predecessor":
                meta["superseded_by"] = ref
                meta["relation"] = relation
            elif direction == "successor":
                meta["replaces"] = ref
                meta["relation"] = relation
            if direction and "tentative" in (relation or "").lower():
                meta["supersession_tentative"] = "True"

            head = prologue(status)
            room = max(CHUNK_CHARS - len(head) - 2, 600)
            pieces = _recursive_split(body, room, ["\n\n", "\n", ". ", " "]) or [body]
            if len(pieces) > 1:
                n_oversized += 1

            for part, piece in enumerate(pieces):
                cid = (f"{doc_id}::{provision}" if len(pieces) == 1
                       else f"{doc_id}::{provision}#p{part}")
                pmeta = dict(meta)
                if len(pieces) > 1:
                    pmeta["part"] = part
                    pmeta["n_parts"] = len(pieces)
                chunks.append(
                    Chunk(text=f"{head}\n{piece}", doc_id=doc_id,
                          chunk_id=cid, meta=pmeta)
                )

    if n_oversized:
        print(f"  {n_oversized} provision(s) longer than {CHUNK_CHARS} chars "
              f"were split into multiple chunks")
    return chunks


def build_index(chunks: list[Chunk], model: str) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415 -- only needed when we actually embed

    INDEX.mkdir(parents=True, exist_ok=True)
    with CHUNKS.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps({
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                "meta": c.meta,
            }, ensure_ascii=False) + "\n")

    print(f"  embedding {len(chunks)} chunks with {model} ...")
    vectors = embed_batch([c.text for c in chunks], model=model,
                          show_progress=True, input_type="passage")
    matrix = np.vstack(vectors).astype("float32")
    np.save(MATRIX, matrix)

    meta = {
        "n_chunks": len(chunks),
        "dim": int(matrix.shape[1]),
        "embedding_model": model,
        "chunk_chars": CHUNK_CHARS,
        "chunks_sha256": hashlib.sha256(CHUNKS.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "n_docs": len({c.doc_id for c in chunks}),
        "n_repealed": sum(1 for c in chunks if c.meta.get("in_force") == "False"),
    }
    INDEX_JSON.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_chunks() -> list[Chunk]:
    if not CHUNKS.exists():
        raise FileNotFoundError(
            f"{CHUNKS} is missing. Run `python -m src.ingest` first."
        )
    out = []
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(Chunk(row["text"], row["doc_id"], row["chunk_id"], row["meta"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None,
                    help="override the EMBED tier (default: profile's EMBED model)")
    ap.add_argument("--stats", action="store_true", help="report on the existing index")
    ap.add_argument("--rebuild", action="store_true", help="re-embed from scratch")
    args = ap.parse_args(argv)

    if args.stats:
        if not INDEX_JSON.exists():
            print("no index yet; run `python -m src.ingest`")
            return 1
        meta = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        chunks = load_chunks()
        roles: dict[str, int] = {}
        for c in chunks:
            roles[c.meta.get("role", "?")] = roles.get(c.meta.get("role", "?"), 0) + 1
        print(f"{meta['n_chunks']} chunks from {meta['n_docs']} documents, "
              f"{meta['dim']}-dim via {meta['embedding_model']}")
        print(f"  current:  {roles.get('current', 0)}")
        print(f"  repealed: {roles.get('repealed', 0)}")
        sizes = sorted(len(c.text) for c in chunks)
        print(f"  chunk chars: min {sizes[0]}, median {sizes[len(sizes)//2]}, "
              f"max {sizes[-1]}")
        multi = sum(1 for c in chunks if c.meta.get("n_parts"))
        print(f"  provisions split across >1 chunk: {multi}")
        return 0

    model = args.model or resolve_model("EMBED")
    chunks = chunk_corpus()
    print(f"chunked {len(chunks)} provisions from "
          f"{len({c.doc_id for c in chunks})} documents")

    if INDEX_JSON.exists() and not args.rebuild:
        existing = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        if (existing.get("embedding_model") == model
                and existing.get("chunks_sha256")
                == hashlib.sha256(CHUNKS.read_bytes()).hexdigest()
                and MATRIX.exists()):
            print("index is already current (use --rebuild to force)")
            return 0

    meta = build_index(chunks, model)
    print(f"wrote {INDEX_JSON.relative_to(REPO)}: {meta['n_chunks']} chunks, "
          f"{meta['dim']}-dim, {meta['n_repealed']} repealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())