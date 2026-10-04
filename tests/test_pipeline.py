"""Tests for the invariants that are cheap to break and expensive to miss.

The emphasis is deliberate. Supersession resolution and the citation contract
both produced wrong-but-plausible output during development -- a repeal warning
naming the wrong section, a valid citation for a question about patents -- and
neither was caught by running the code. They were caught by asserting against
the index. So that is what these tests do.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aip.evals import retrieval_metrics
from src.build import _heading_match, supersession_index
from src.retrieve import load_golden

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "data" / "eval" / "golden.jsonl"


@pytest.fixture(scope="module")
def index_meta() -> dict[str, dict]:
    meta = {}
    path = REPO / "data" / "index" / "chunks.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)["meta"]
            meta.setdefault(entry["provision"], entry)
    return meta


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return load_golden(GOLDEN)


# --------------------------------------------------------------------------
# Supersession
# --------------------------------------------------------------------------
def test_weak_links_are_never_stated_as_fact(index_meta: dict[str, dict]) -> None:
    """Where the map grades its own row `weak`, the corpus must hedge it.

    Not a heading-equality test. Requiring the resolved counterpart's heading to
    match would fail ~600 links that are correct but reworded -- "Classes
    Criminal Courts" really does become "Classes of Criminal Courts" -- so a
    heading test would either be uselessly strict or need a threshold nobody can
    defend. What matters is narrower: a `weak` successor must not be stated as
    settled.

    Scoped to provisions actually in the corpus. The manifest selects sections,
    not whole Acts, so most `weak` links belong to sections that were never
    fetched and have nothing to render.
    """
    index = supersession_index()
    # Chunk meta has no `number`; the provision label is "CrPC s.10", so the
    # number is the tail after the last " s." (act labels contain spaces:
    # "Evidence Act s.1").
    built = {
        f"{meta['act']}::{meta['provision'].rsplit(' s.', 1)[-1]}"
        for meta in index_meta.values()
    }
    weak = [key for key, rec in index.items()
            if rec["relation"] == "weak" and key in built]

    corpus = "\n".join(
        p.read_text(encoding="utf-8") for p in (REPO / "data" / "corpus").glob("*.md")
    )
    unhedged = [key for key in weak if not _section_hedged(key, corpus)]
    assert not unhedged, f"weak successor links stated without a caveat: {unhedged[:5]}"

    # Every weak link the corpus does render must carry the hedge, and the hedge
    # must not be silently dropped by a refactor that changes rel_txt.
    hedges = sum(1 for key in weak if _section_hedged(key, corpus))
    assert hedges == len(weak)
    assert "tentative" in corpus


_LABELS = {"bns": "BNS", "bnss": "BNSS", "bsa": "BSA",
           "ipc": "IPC", "crpc": "CrPC", "iea": "Evidence Act"}


def _section_hedged(key: str, corpus: str) -> bool:
    """Does this provision's own block carry the tentative caveat?"""
    act, number = key.split("::")
    anchor = f"## {_LABELS.get(act, act)} s.{number} - "
    start = corpus.find(anchor)
    if start < 0:
        return False
    end = corpus.find("\n## ", start)
    return "tentative" in corpus[start:end if end > 0 else len(corpus)]


@pytest.mark.parametrize("current,section", [
    ("bns", "103"), ("bns", "318"), ("bns", "126"), ("bns", "127"),
    ("bnss", "72"), ("bnss", "117"), ("bnss", "239"), ("bnss", "528"),
    ("bsa", "55"), ("bsa", "63"), ("bsa", "39"),
])
def test_known_repeal_pairs(current: str, section: str) -> None:
    """Spot-check the pairs a lawyer would recognise by sight.

    Not a golden table: a spot-check that would pass if the resolver were right
    for the wrong reason. Each pair was verified by reading both headings.
    """
    expected = {
        "bns:103": "ipc:302", "bns:318": "ipc:415", "bns:126": "ipc:339",
        "bns:127": "ipc:340", "bnss:72": "crpc:70", "bnss:117": "crpc:105E",
        "bnss:239": "crpc:216", "bnss:528": "crpc:482",
        "bsa:55": "iea:60", "bsa:63": "iea:65B", "bsa:39": "iea:45",
    }
    index = supersession_index()
    record = index.get(f"{current}::{section}")
    assert record is not None
    got = f"{record['act']}:{record['section']}"
    assert got == expected[f"{current}:{section}"]


def test_heading_match_prefers_the_exact_heading() -> None:
    assert _heading_match("Grievous hurt", "Grievous hurt") == 2
    assert _heading_match("Punishment for hurt", "Hurt") == 1
    assert _heading_match("Murder", "Cheating") == 0


def test_every_repealed_provision_names_its_replacement(index_meta: dict) -> None:
    """A repealed provision with no successor is a dead end for the model."""
    orphans = [
        provision for provision, meta in index_meta.items()
        if meta.get("in_force") == "False" and not meta.get("superseded_by")
    ]
    # The 2024 repeal is corpus-wide, so any gap here is a resolver bug.
    assert not orphans, f"repealed provisions with no counterpart: {orphans[:10]}"


# --------------------------------------------------------------------------
# Golden set
# --------------------------------------------------------------------------
def test_golden_is_valid(cases: list[dict]) -> None:
    from src.validate_golden import main as validate_main

    assert validate_main() == 0


def test_every_case_cites_an_existing_provision(cases: list[dict],
                                                index_meta: dict) -> None:
    for case in cases:
        for ref in case["relevant_provisions"] + case["forbidden_provisions"]:
            assert ref in index_meta, f"{case['id']} cites unknown provision {ref}"


def test_current_law_cases_do_not_cite_repealed_text(cases: list[dict],
                                                    index_meta: dict) -> None:
    for case in cases:
        if case["category"] == "historical":
            continue
        for ref in case["relevant_provisions"]:
            assert index_meta[ref]["in_force"] != "False", (
                f"{case['id']} asks current law but cites repealed {ref}"
            )


def test_unanswerable_cases_cite_nothing(cases: list[dict]) -> None:
    for case in cases:
        if case["expect_refusal"]:
            assert not case["relevant_provisions"], case["id"]
            assert not case["forbidden_provisions"], case["id"]


def _short_of(ref: str, index_meta: dict[str, dict]) -> str:
    """Act label for a citation, matched by longest prefix.

    Splitting on ' s.' and taking [0] gives 'Arbitration' where the label is
    'Arbitration Act', which silently under-reports coverage.
    """
    best = ""
    for meta in index_meta.values():
        if ref.startswith(f"{meta['act_short']} ") and len(meta["act_short"]) > len(best):
            best = meta["act_short"]
    return best


def test_golden_covers_every_document(index_meta: dict, cases: list[dict]) -> None:
    cited = {
        _short_of(ref, index_meta)
        for case in cases
        for ref in case["relevant_provisions"]
    }
    present = {meta["act_short"] for meta in index_meta.values()}
    missing = present - cited
    assert not missing, f"corpus documents never exercised by the golden set: {missing}"


def test_golden_has_a_trap_in_every_recodified_act(cases: list[dict]) -> None:
    """The point of the corpus is the old/new pair, so every pair needs a test."""
    trapped = set()
    for case in cases:
        if case["forbidden_provisions"]:
            for ref in case["forbidden_provisions"]:
                short = ref.split(" s.")[0]
                if short in {"IPC", "CrPC", "Evidence Act"}:
                    trapped.add(short)
    assert {"IPC", "CrPC", "Evidence Act"} <= trapped


# --------------------------------------------------------------------------
# Metric plumbing
# --------------------------------------------------------------------------
def test_retrieval_metrics_key_names() -> None:
    """`retrieval_metrics` takes ks= and returns hit_rate@k.

    This is asserted because the caller previously passed k= and read hit@k,
    which is a TypeError at best and a silent zero at worst.
    """
    out = retrieval_metrics(["a", "b"], ["b"], ks=(1, 5))
    assert "hit_rate@1" in out and "hit_rate@5" in out
    assert out["hit_rate@1"] == 0.0
    assert out["hit_rate@5"] == 1.0
    assert out["mrr"] == 0.5


def test_retrieval_metrics_multi_relevant(cases: list[dict]) -> None:
    multi = [c for c in cases if len(c["relevant_provisions"]) > 1]
    assert multi, "the golden set should contain multi-hop cases"
    gold = multi[0]["relevant_provisions"]
    assert retrieval_metrics(gold[:1], gold, ks=(1,))["recall@1"] < 1.0


def test_citation_markers_are_strict() -> None:
    """Only [1] is a citation. [see 1] and [1,2] are not."""
    from src.answer import _CITE

    assert _CITE.findall("text [1] more [2]") == ["1", "2"]
    assert _CITE.findall("see [1] and [note 2]") == ["1"]
    assert _CITE.findall("no citations here") == []


def test_out_of_scope_terms_are_not_over_broad() -> None:
    """Regression: 'judgment or award' refused a Motor Vehicles question."""
    from src.answer import _OUT_OF_SCOPE

    for fine in [
        "What is the duty of an insurer to satisfy a judgment or award?",
        "What is the punishment for murder?",
        "What is cheating by personation?",
    ]:
        assert not _OUT_OF_SCOPE.search(fine), fine

    for out in [
        "What is the penalty for late filing of a patent application?",
        "What did the Supreme Court hold in the Kesavananda Bharati case?",
        "What is the current corporate tax rate?",
    ]:
        assert _OUT_OF_SCOPE.search(out), out