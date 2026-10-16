"""Two cache layers, and an honest account of what the second one costs.

The trap this module is built around
------------------------------------
A semantic cache looks like free performance. You return a stored answer when a
new question is close enough to an old one, and latency and cost fall while
nothing errors. But cosine similarity does not know what a question is *about*.
On this corpus:

    "What is the punishment for murder under the BNS?"      cosine ~0.97
    "What is the punishment for murder under the IPC?"      <- different answer

Both are about murder; they are about different *law*. Serve the first for the
second and you have returned a confident, well-cited answer to a question nobody
asked, pointing at a provision repealed on 1 July 2024. That is the exact
failure this project exists to prevent, arriving through the performance
optimisation.

So the threshold is the deliverable, not the cache. `src/cache_sweep.py` measures
it: it sweeps the threshold over the golden set and reports where wrong hits
begin. What it found is stronger than a number -- **there is no safe similarity
band on this corpus at all.** Across 9,702 ordered question pairs the maximum
off-diagonal cosine is 0.8699, and every pair above 0.80 is a wrong hit:

    "When may a temporary injunction be granted?"   cosine 0.82
    "When is a perpetual injunction be granted?"    <- different answer

That is the Gold/Silver example in statute form, and it is why the layer ships
above the measured maximum rather than at a number that merely looks cautious.

Two layers
----------
**Exact.** Normalised question plus retrieval configuration. Identical question,
identical answer, by construction -- there is no similarity judgement to get
wrong. This layer also serves cached *refusals*, which is correct for an
identical question.

**Semantic.** Cosine similarity over question embeddings, above the measured
threshold, with a scope guard. It never serves a cached refusal: a refusal is a
decision about one question, and propagating it to a merely-similar question
converts a correct refusal into a wrong one, which is harder to notice.

The scope guard
---------------
Cosine similarity is blind to which statute is being asked about, and on a
corpus holding both generations of Indian criminal law that is the single most
dangerous axis there is. So a semantic hit additionally requires the two
questions to agree on:

  * which Act they name (BNS vs IPC, BNSS vs CrPC, BSA vs Evidence Act, ...)
  * which numbered provision they cite ("section 138" vs "section 139")

If either set is disjoint, no hit, at any similarity. This is deliberately a
*conservative* guard: it only ever blocks hits, so it cannot introduce a wrong
answer, and it raises the hit rate without pretending to solve semantic
equivalence in general.

What it does not do
-------------------
It does not distinguish "Gold" from "Silver". Two questions differing only in a
bare entity name, with no Act and no section number, are lexically almost
identical and semantically opposite, and no threshold plus no token guard can
tell them apart from the question text alone.

The sweep says this is not a hypothetical here: it is the dominant failure, and
the guard catches **none** of the observed pairs, because "temporary" versus
"perpetual" injunction, or BNSS s.72 versus s.41, share an Act and a provision
number. The guard blocks by axis, and on this corpus the dangerous axis is the
adjective, which no lexical guard sees.

That is why the honest engineering answer was not "tune the guard" but "measure
the band, find it empty, and ship the layer where it cannot fire." The guard
stays because it costs nothing and blocks the Act-identity collisions that a
wider question log would eventually produce -- but it is not what saves this
system, and the report does not claim that it is.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aip.embed import embed

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO / "data" / "cache"
DB_PATH = CACHE_DIR / "answers.sqlite3"

# Measured, not reasoned. `python -m src.cache_sweep` sweeps the threshold over
# 9,702 ordered pairs of golden questions and reports precision at every step.
#
# What it found, and it is the headline result of this module:
#
#   * the maximum off-diagonal similarity across all 9,702 pairs is 0.8699
#   * at every threshold with any hits at all, the hit rate is 0.000 precision
#     -- every similar pair on this corpus is a pair of questions about
#     *different* provisions
#   * above 0.87 there are no hits whatsoever
#
# So there is no similarity band on this corpus that is safe, and the correct
# shipping value is one above the measured maximum, where the layer is inert.
# 0.95 leaves headroom above 0.8699 for a corpus with a slightly wider band.
#
# The intuition-driven 0.95 is the right *ballpark* and the wrong *reason*: it
# would still have been unsafe on a corpus whose questions collide lower. The
# number is an experiment result and the experiment ships next to it, in
# data/eval/cache_sweep.json.
DEFAULT_SEMANTIC_THRESHOLD = 0.95

# Which Act is being asked about. The canonical token plus the spellings that
# actually occur, because "Indian Penal Code" and "IPC" must map to the same
# axis or the guard is theatre.
_ACT_ALIASES: dict[str, tuple[str, ...]] = {
    "BNS": ("bns", "bharatiya nyaya sanhita", "nyaya sanhita"),
    "IPC": ("ipc", "indian penal code", "penal code"),
    "BNSS": ("bnss", "bharatiya nagarik suraksha sanhita",
             "nagarik suraksha sanhita", "bharatiya nagarik suraksha sankhita"),
    "CrPC": ("crpc", "code of criminal procedure", "criminal procedure code"),
    "BSA": ("bsa", "bharatiya sakshya adhiniyam", "sakshya adhiniyam",
            "sakshya adhiniyam"),
    "EvidenceAct": ("evidence act", "indian evidence act"),
    "NIAct": ("ni act", "negotiable instruments act"),
    "CPC": ("cpc", "code of civil procedure", "civil procedure code"),
    "ITAct": ("it act", "information technology act"),
}

_ACT_TOKENS: tuple[str, ...] = tuple(
    alias for aliases in _ACT_ALIASES.values() for alias in aliases
)

# A provision number is a hard identity: s.138 and s.139 of the NI Act are
# different offences with different penalties. Keep the letter suffix, because
# "124A" and "124" are not the same provision.
_NUMBER = re.compile(r"\b\d+[a-z]?\b")

_WS = re.compile(r"\s+")


def normalise(question: str) -> str:
    """Fold a question to its comparison form.

    Case, surrounding whitespace, internal runs of whitespace, and trailing
    punctuation only. Nothing semantic: stripping words would merge questions
    that differ exactly where it matters.
    """
    return _WS.sub(" ", question.strip().lower()).strip("?.! ,;:")


def scope_of(question: str) -> dict[str, set[str]]:
    """The two axes a hit must agree on: named Act, and cited provision number."""
    low = question.lower()
    acts = {name for name, aliases in _ACT_ALIASES.items()
            if any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in aliases)}
    return {"acts": acts, "numbers": set(_NUMBER.findall(low))}


def scopes_compatible(cached: dict[str, set[str]],
                      probe: dict[str, set[str]]) -> tuple[bool, str]:
    """May a cached answer be served for this probe?

    Returns (allowed, reason). The reason is kept because a cache that silently
    declines to fire is impossible to tune and a cache that silently fires wrongly
    is the thing this whole module exists to prevent.
    """
    if not cached["acts"] or not probe["acts"]:
        # One side names no Act at all. Absence is not agreement, so this is
        # allowed -- but it is the case the guard cannot help with, and the
        # caller may want to know it happened.
        pass
    elif not (cached["acts"] & probe["acts"]):
        return False, (f"different Acts: cached "
                       f"{sorted(cached['acts'])} vs probe {sorted(probe['acts'])}")
    elif cached["acts"] != probe["acts"]:
        # Overlapping but not identical: "BNS" and "BNS and IPC" are different
        # questions, and the supersession corpus makes that the dangerous one.
        return False, (f"partial Act overlap: cached "
                       f"{sorted(cached['acts'])} vs probe {sorted(probe['acts'])}")

    if (cached["numbers"] or probe["numbers"]) and cached["numbers"] != probe["numbers"]:
        return False, (f"different section numbers: cached "
                       f"{sorted(cached['numbers'])} vs probe "
                       f"{sorted(probe['numbers'])}")
    return True, ""


def _config_key(config: Config) -> str:
    """Canonical form of a retrieval config, used as the semantic-layer mask."""
    return json.dumps(config.as_dict(), sort_keys=True)


@dataclass
class Config:
    """Everything that changes the answer for a fixed question.

    Part of the cache key for a reason that is easy to get wrong: the same
    question against `current_only=false` retrieves repealed law and gets a
    different answer, and against `k=3` instead of `k=8` gets a different
    citation set. Keying on the question alone would serve a current-law answer
    to someone who explicitly asked for the old one.

    `mode` is the generator: `extractive`, `llm`, or `agent`. An agent answer is
    produced by a different process with a different citation set, so serving it
    to an extractive caller (or vice versa) is a wrong answer that looks right.
    """

    retriever: str = "hybrid+current"
    current_only: bool = True
    k: int = 8
    mode: str = "extractive"

    def as_dict(self) -> dict[str, Any]:
        return {"retriever": self.retriever, "current_only": self.current_only,
                "k": self.k, "mode": self.mode}


@dataclass
class Lookup:
    """The result of a cache probe, including the reason it went the way it did."""

    hit: bool
    layer: str = "none"          # "exact" | "semantic" | "none"
    similarity: float | None = None
    matched_question: str | None = None
    reason: str = ""
    payload: dict[str, Any] | None = None


@dataclass
class CacheStats:
    """Counters for /metrics. `blocked_scope` is the interesting one."""

    exact_hits: int = 0
    semantic_hits: int = 0
    blocked_scope: int = 0
    below_threshold: int = 0
    misses: int = 0
    stores: int = 0

    def as_dict(self) -> dict[str, Any]:
        total = self.exact_hits + self.semantic_hits + self.misses
        return {
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "blocked_scope": self.blocked_scope,
            "below_threshold": self.below_threshold,
            "misses": self.misses,
            "stores": self.stores,
            "lookups": total,
            "hit_rate": round((self.exact_hits + self.semantic_hits) / total, 4)
            if total else 0.0,
        }


class AnswerCache:
    """SQLite-backed exact + semantic answer cache.

    Disk rather than a dict because a service cache that forgets everything on
    restart cannot be demoed on a second day, and because the hit rate is one of
    the numbers in the report. Write-through: a miss stores immediately, so the
    next identical question is an exact hit.

    The embedding matrix is held in memory and rebuilt from SQLite on first use,
    because a linear scan over a few thousand rows of 384 floats is far cheaper
    than a query per lookup and this corpus is small enough for it to be safe.
    """

    def __init__(self, path: Path | None = None, *,
                 semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
                 enable_semantic: bool = True) -> None:
        self.path = path or DB_PATH
        self.semantic_threshold = semantic_threshold
        self.enable_semantic = enable_semantic
        self._lock = threading.Lock()
        self.stats = CacheStats()
        self._matrix: np.ndarray | None = None
        self._vectors: list[np.ndarray] = []
        self._index: list[sqlite3.Row] = []
        self._configs: list[str] = []
        self._create_schema()

    # -- storage ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """A raw connection. Callers must close it; prefer `self._db()`.

        Left public only because `_create_schema` needs a connection it can
        close explicitly. Every other caller goes through `_db`, which closes.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        # Named access throughout; the default row factory returns bare tuples,
        # which fails at `row["payload"]` rather than at the obvious place.
        conn.row_factory = sqlite3.Row
        return conn

    @contextlib.contextmanager
    def _db(self) -> Any:
        """Open, yield, commit, close.

        Two defects live here if you write it the obvious way. `with
        sqlite3.connect(...)` does NOT close the connection -- it commits the
        transaction and leaves the handle open -- so a per-request cache leaks
        one handle per query and eventually holds SQLite locks that make a
        subsequent reader see stale state. And running `CREATE TABLE IF NOT
        EXISTS` on every connect puts DDL on the hot path for no reason; the
        schema is created once, in `_create_schema`.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_schema(self) -> None:
        with self._db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS answers (
                    exact_key   TEXT PRIMARY KEY,
                    question    TEXT NOT NULL,
                    config      TEXT NOT NULL,
                    payload     TEXT NOT NULL,
                    refused     INTEGER NOT NULL DEFAULT 0,
                    scope       TEXT NOT NULL,
                    vector      BLOB,
                    dim         INTEGER,
                    created_at  REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_answers_config "
                         "ON answers(config)")

    @staticmethod
    def _exact_key(question: str, config: Config) -> str:
        blob = json.dumps(
            {"q": normalise(question), "cfg": config.as_dict()},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _load_matrix(self, desired_dim: int | None = None) -> None:
        if self._matrix is not None and (desired_dim is None
                                       or self._matrix.shape[1] == desired_dim):
            return
        rows: list[sqlite3.Row] = []
        vectors: list[np.ndarray] = []
        with self._db() as conn:
            rows = conn.execute(
                "SELECT exact_key, question, refused, scope, vector, dim, config "
                "FROM answers WHERE vector IS NOT NULL"
            ).fetchall()
        # Ignore embeddings produced by another embedding model/dimension. This
        # makes a mixed cache loadable instead of crashing the whole request.
        target_dim = desired_dim
        if target_dim is None and rows:
            for candidate in rows:
                if candidate["dim"] is not None:
                    target_dim = candidate["dim"]
                    break
        filtered_rows: list[sqlite3.Row] = []
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if row["dim"] is None:
                continue
            if target_dim is not None and row["dim"] != target_dim:
                continue
            if target_dim is not None and vector.shape[0] != target_dim:
                continue
            filtered_rows.append(row)
            vectors.append(vector)
        self._index = filtered_rows
        self._vectors = vectors
        self._configs = [row["config"] for row in filtered_rows]
        self._matrix = (np.vstack(vectors) if vectors
                        else np.zeros((0, 0), dtype=np.float32))

    # -- lookups ---------------------------------------------------------
    def lookup(self, question: str, config: Config,
               vector: np.ndarray | None = None) -> Lookup:
        """Probe both layers. Exact first, then semantic, then miss."""
        key = self._exact_key(question, config)
        with self._lock:
            with self._db() as conn:
                row = conn.execute(
                    "SELECT payload FROM answers WHERE exact_key = ?", (key,)
                ).fetchone()
            if row is not None:
                self.stats.exact_hits += 1
                return Lookup(hit=True, layer="exact", similarity=1.0,
                              matched_question=question, reason="exact key match",
                              payload=json.loads(row["payload"]))

            if not self.enable_semantic:
                self.stats.misses += 1
                return Lookup(hit=False, reason="semantic layer disabled")

            probe_scope = scope_of(question)
            probe_vec = vector if vector is not None else self.embed_question(question)
            self._load_matrix(desired_dim=probe_vec.shape[0])
            if self._matrix is None or len(self._vectors) == 0:
                self.stats.misses += 1
                return Lookup(hit=False, reason="semantic cache empty")

            if self._matrix.shape[1] != probe_vec.shape[0]:
                # The stored vectors were produced by a different embedding
                # model -- a profile change, or a cache copied from another
                # machine. A cosine against them is meaningless, so the layer
                # is reset rather than serving a number computed from
                # mismatched dimensions.
                self.stats.misses += 1
                return Lookup(hit=False,
                              reason="embedding dimension changed; semantic cache reset")
            # Vectors are L2-normalised on the way in, so a dot product is the
            # cosine. Same embedding model as the corpus, by construction.
            scores = self._matrix @ probe_vec

            # The retrieval configuration gates the semantic layer too, not just
            # the exact key. The same question asked with current_only=False is a
            # *different question* -- it reaches repealed provisions and must not
            # be answered with a current-law answer recorded against the filtered
            # request. Without this mask the semantic layer silently reintroduces
            # the bug the current-law filter exists to prevent.
            want_config = _config_key(config)
            if self._configs:
                eligible = np.array(
                    [c == want_config for c in self._configs], dtype=bool
                )
                if not eligible.any():
                    self.stats.misses += 1
                    return Lookup(hit=False,
                                  reason="no cached entry for this configuration")
                scores = np.where(eligible, scores, -1.0)

            best_i = int(np.argmax(scores))
            best = float(scores[best_i])

            if best < self.semantic_threshold:
                self.stats.below_threshold += 1
                self.stats.misses += 1
                return Lookup(hit=False, similarity=round(best, 4),
                              reason=(f"best similarity {best:.4f} < "
                                      f"{self.semantic_threshold}"))

            candidate = self._index[best_i]
            cached_scope = {k: set(v) for k, v in
                            json.loads(candidate["scope"]).items()}
            allowed, why = scopes_compatible(cached_scope, probe_scope)
            if not allowed:
                self.stats.blocked_scope += 1
                self.stats.misses += 1
                return Lookup(hit=False, similarity=round(best, 4),
                              reason=f"scope guard blocked: {why}")

            if candidate["refused"]:
                # A refusal is a decision about one specific question. Serving
                # it to a merely-similar question turns a correct refusal into a
                # wrong answer, which is harder to notice than a missing one.
                self.stats.misses += 1
                return Lookup(hit=False, similarity=round(best, 4),
                              reason="cached entry is a refusal; not served")

            with self._db() as conn:
                row = conn.execute("SELECT payload FROM answers WHERE exact_key = ?",
                                   (candidate["exact_key"],)).fetchone()
            self.stats.semantic_hits += 1
            return Lookup(hit=True, layer="semantic", similarity=round(best, 4),
                          matched_question=candidate["question"],
                          reason=f"cosine {best:.4f} >= {self.semantic_threshold}",
                          payload=json.loads(row["payload"]) if row else None)

    # -- stores ----------------------------------------------------------
    def store(self, question: str, config: Config, payload: dict[str, Any],
              *, refused: bool | None = None,
              vector: np.ndarray | None = None) -> None:
        """Remember an answer. Refusals are stored; they are just not served
        to anything other than an exact match.

        `refused` defaults to None and is then read out of the payload, because
        the alternative -- a caller forgetting to pass it -- marks a refusal as
        an answer, and the semantic layer will then hand it to the next similar
        question. Deriving it from the payload cannot drift from the payload.
        """
        if not payload.get("citations_valid", True):
            # Never cache an answer that failed its own contract. Replaying it
            # would make a transient bug permanent.
            return
        is_refusal = bool(payload.get("refused")) if refused is None else refused
        # Exact caching needs only the normalized question key. Do not pay for
        # an embedding when semantic lookup is disabled; the shipped corpus has
        # no empirically safe semantic-hit band, so the service disables that
        # layer by default.
        vec32: np.ndarray | None = None
        if vector is not None or self.enable_semantic:
            vec = vector if vector is not None else self.embed_question(question)
            vec32 = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(vec32))
            if norm:
                vec32 = vec32 / norm
        key = self._exact_key(question, config)
        scope = scope_of(question)
        with self._lock:
            with self._db() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO answers "
                    "(exact_key, question, config, payload, refused, scope, "
                    " vector, dim, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (key, question, _config_key(config),
                     json.dumps(payload, ensure_ascii=False), int(is_refusal),
                     json.dumps({k: sorted(v) for k, v in scope.items()}),
                     vec32.tobytes() if vec32 is not None else None,
                     int(vec32.shape[0]) if vec32 is not None else None,
                     time.time()),
                )
            self.stats.stores += 1
            # Keep the in-memory matrix consistent with what is on disk.
            self._matrix = None
            self._vectors = []
            self._index = []
            self._configs = []

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def embed_question(question: str, model: str | None = None) -> np.ndarray:
        # `input_type="query"` matters on asymmetric models (NVIDIA NIM): a
        # passage vector and a query vector for the same text differ, and using
        # the wrong one quietly halves the hit rate.
        return embed(question, model=model, input_type="query")

    def size(self) -> int:
        with self._db() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0])

    def clear(self) -> int:
        with self._lock:
            with self._db() as conn:
                cur = conn.execute("DELETE FROM answers")
                removed = cur.rowcount
            self._matrix = None
            self._vectors = []
            self._index = []
            self._configs = []
            self.stats = CacheStats()
        return removed


_DEFAULT: AnswerCache | None = None
_DEFAULT_LOCK = threading.Lock()


def get_cache(path: Path | None = None, *,
              semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
              enable_semantic: bool = True) -> AnswerCache:
    """The process-wide cache. Built once, because building it per request is
    the kind of thing that looks free and shows up in the p95."""
    global _DEFAULT
    if path is not None:
        return AnswerCache(path, semantic_threshold=semantic_threshold,
                           enable_semantic=enable_semantic)
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = AnswerCache(semantic_threshold=semantic_threshold,
                                   enable_semantic=enable_semantic)
    return _DEFAULT


def reset_default() -> None:
    """Drop the process-wide cache. Used by tests."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = None
