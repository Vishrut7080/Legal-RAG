"""Normalise the raw snapshot into a corpus that a chunker can be pointed at.

    python -m src.build            # data/raw/ -> data/corpus/
    python -m src.build --stats    # what is in the corpus, and its shape

Output is one Markdown file per Act plus `data/corpus/index.json`.

Why Markdown, and why one file per Act
--------------------------------------
Markdown is a container, not a format. What matters is that the unit a human
reads and the unit the chunker splits are the same unit: a provision. A
retriever that ranks "IPC s.302" and then hands the model a window of unrelated
text is not retrieval, it is a lucky coincidence.

So each provision gets an `##`-headed block whose heading line is exactly the
citation a lawyer would write -- `## BNS s.103 - Punishment for murder` -- and
whose body opens with a compact status block. That block is not decoration:

    - In force: yes
    - Replaced: IPC s.302 (relation near-identical, score 0.90)

Two reasons it earns its place.

1. It makes the supersession trap *visible at the chunk level*. A model that
   can see "this provision replaced IPC s.302" can be told to prefer it, and a
   retrieval metric can be checked for having returned a repealed provision.
2. It puts "in force" into the embedded text. Metadata is invisible to an
   embedding model; this is not.

Licence discipline
------------------
The snapshot deliberately holds only legislative text. Section bodies,
headings, and the Schedule-derived offence classification (cognizable /
bailable / triable by) are kept. The HTML rendering, the editorial `note`
field, and judgment holdings are dropped -- the upstream corpus is published
under s.52(1)(q)(ii), Copyright Act 1957, which exempts Acts from copyright
but not the commentary around them. Only the number of judgments citing a
provision is carried, which is a fact about the corpus, not a reproduction of
it. Do not widen this without checking that.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections.abc import Iterator  # noqa: F401  (used in type hints below)
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
CORPUS = REPO / "data" / "corpus"
MANIFEST = REPO / "data" / "manifest.json"

# Amendment history is appended inline to section text as a run of numbered
# notes ("1. Ins. by Act 46 of 1999, s. 17 (w.e.f. 1-7-2002). 2. Subs. by ...").
# It is legitimate legislative material, but it is dense citation boilerplate
# that dominates the embedding of any provision that has been amended often.
# Split it off, keep it, and let build.py mark the provision.
_NOTE_RUN = re.compile(
    r"(?:\s*\b\d+\.\s+(?:Ins\.|Subs\.|Rep\.|Ren\.|Omitted|Added|Mod\.).{10,}?)+",
    re.IGNORECASE | re.DOTALL,
)

_WS = re.compile(r"[ \t]+")


def split_notes(text: str) -> tuple[str, str]:
    """Return (operative text, amendment notes)."""
    m = _NOTE_RUN.search(text)
    if not m:
        return text.strip(), ""
    return text[: m.start()].strip(), m.group().strip()


def clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


# --------------------------------------------------------------------------
# Reading the snapshot
# --------------------------------------------------------------------------
def read_raw(relpath: str) -> dict[str, Any]:
    path = RAW / relpath
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m src.fetch` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def act_metadata(act_id: str) -> dict[str, Any]:
    return read_raw(f"acts/{act_id}.json")["act"]


def _heading_map() -> dict[str, str]:
    """'act::number' -> upstream heading, for every section of every fetched Act.

    Read from the raw Act metadata, which lists all upstream sections with their
    headings. No extra request is needed.
    """
    out: dict[str, str] = {}
    for path in sorted((RAW / "acts").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        act = payload.get("act") or {}
        # The section index sits at the top level of the payload, not under "act".
        for section in payload.get("sections") or []:
            out[f"{act.get('id')}::{section.get('number')}"] = clean(
                section.get("heading") or ""
            )
    return out


def _heading_match(left: str, right: str) -> int:
    """Grade how well two provision headings correspond: 2 same, 1 subset, 0 other.

    This is the tie-break that makes the correspondence usable. The upstream
    map is a similarity join, not a legal cross-reference table, and it
    confuses a definition section with the punishment section that follows it,
    because the two headings differ only by a leading "Punishment for". Left to
    score alone it resolves BNS s.115 "Punishment for voluntarily causing hurt"
    to IPC s.321 "Voluntarily causing hurt" -- the definition -- instead of IPC
    s.323 "Punishment for voluntarily causing hurt", and likewise BNS s.126 to
    IPC s.339 rather than IPC s.341. A supersession warning that names the
    wrong section is worse than no warning, so headings outrank score.
    """
    a, b = left.strip().lower().rstrip("."), right.strip().lower().rstrip(".")
    if not a or not b:
        return 0
    if a == b:
        return 2
    if a in b or b in a:
        return 1
    return 0


def _edge_rank(
    edge: tuple[str, str, str, str, str, str, float],
    act: str,
    own_heading: str,
    headings: dict[str, str],
    rank: dict[str, int],
) -> tuple[int, int, float]:
    """Order candidate counterparts: heading match, then relation, then score."""
    # The counterpart is on whichever end of the edge this key is not on.
    other = f"{edge[2]}::{edge[3]}" if edge[0] == act else f"{edge[0]}::{edge[1]}"
    return (
        _heading_match(own_heading, headings.get(other, "")),
        rank.get(edge[5] or "", 0),
        edge[6] or 0.0,
    )


def supersession_index() -> dict[str, dict[str, Any]]:
    """Map 'act::number' -> counterpart record, from the correspondence maps.

    Built once and applied while rendering, so that the "replaced by" line on a
    BNS provision and the "repealed, replaced by" line on its IPC counterpart
    are generated from a single source and cannot drift apart.

    The maps are many-to-one, and that is a property of the law rather than a
    defect in the data: BNS folds several 1860 offences into one section, so
    IPC s.323 and s.324 both land in BNS s.108, and BNS s.318 gathers IPC s.415,
    s.418 and s.420. Taking the last row wins therefore links BNS s.318 to the
    *worst* candidate, which silently turns a repeal warning into the wrong
    repeal warning. So: build the edge set once, then resolve each key against
    its own candidate list, ranked by heading match, then relation strength,
    then score.

    Each key resolves independently, which means the relation is not symmetric
    and is not meant to be: where one current section absorbs several repealed
    ones, only one of them can be named as *the* counterpart. Symmetry was tried
    here and reverted -- requiring `A -> B` to imply `B -> A` cost coverage,
    because the losing side of a merged section was left with no counterpart at
    all, and "this 1860 section has no successor" is a worse thing to tell a
    user than "this 1860 section was folded into BNS s.108, whose own mapping
    names a different predecessor". Coverage wins; the loser's confidence is
    recorded in `relation` and `score` instead.
    """
    rank = {"near-identical": 3, "close": 2, "weak": 1}
    headings = _heading_map()

    # act::number -> every edge touching it, from either side.
    candidates: dict[str, list[tuple[str, str, str, str, str, str, float]]] = {}
    for pair_file in sorted((RAW / "mappings").glob("*.json")):
        blob = json.loads(pair_file.read_text(encoding="utf-8"))
        for row in blob.get("rows", []):
            fa, fs = row.get("from_act"), row.get("from_section")
            ta, ts = row.get("to_act"), row.get("to_section")
            if not (fa and fs and ta and ts):
                continue
            edge = (fa, fs, ta, ts, blob.get("pair"), row.get("relation"), row.get("score"))
            for key in (f"{fa}::{fs}", f"{ta}::{ts}"):
                candidates.setdefault(key, []).append(edge)

    out: dict[str, dict[str, Any]] = {}
    for key, edges in candidates.items():
        act, _number = key.split("::")
        fa, fs, ta, ts, pair, relation, score = max(
            edges, key=lambda e: _edge_rank(e, act, headings.get(key, ""), headings, rank)
        )
        out[key] = {
            "act": ta if act == fa else fa,
            "section": ts if act == fa else fs,
            "relation": relation,
            "score": score,
            "pair": pair,
        }
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render_section(
    act: dict[str, Any],
    label: str,
    payload: dict[str, Any],
    supersession: dict[str, dict[str, Any]],
    labels: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    section = payload.get("section", {})
    number = str(section.get("number"))
    heading = clean(section.get("heading")) or f"Section {number}"
    operative, notes = split_notes(clean(section.get("text")))

    ref = f"{label} s.{number}"
    counterpart = supersession.get(f"{act['id']}::{number}")

    in_force = act.get("in_force")
    lines: list[str] = [f"## {ref} - {heading}", ""]
    lines.append(f"- Provision: {ref}")
    lines.append(f"- Heading: {heading}")
    lines.append(f"- In force: {'yes' if in_force else 'no (repealed)'}")
    if counterpart:
        other_id = counterpart["act"]
        other = f"{labels.get(other_id, other_id)} s.{counterpart['section']}"
        rel = counterpart.get("relation")
        rel_txt = f"{rel}, score {counterpart.get('score')}"
        # The upstream map grades its own rows, and 41 of the 2162 resolved links
        # come back `weak` -- a similarity join that never found the real
        # counterpart, e.g. CrPC s.10 "Subordination of Assistant Sessions
        # Judges" scoring 0.61 against BNSS s.8 "Court of Session". Most weak
        # links are still right (IPC s.435 "Mischief by fire or explosive
        # substance" does land in BNS s.326) and dropping them all would lose
        # real information, so they are kept and marked instead. A repeal
        # warning that overstates its own confidence is how a user ends up
        # citing the wrong successor as settled.
        tentative = rel == "weak"
        if tentative:
            rel_txt += " - tentative, confirm against the commencement notification"
        if in_force:
            lines.append(f"- Replaced: {other} ({rel_txt})")
        else:
            # Two colon-delimited keys rather than one prose sentence:
            # `src/ingest.py` parses this block back into chunk metadata, and a
            # key that has to be recovered out of free text is a key that will
            # silently stop being recovered one day.
            lines.append("- Repealed on: 2024-07-01")
            lines.append(f"- Replaced by: {other} ({rel_txt})")
    else:
        tentative = False
        lines.append("- Replaced by: none in the recodification maps")

    for row in payload.get("classification") or []:
        bits = [
            f"{k}={v}" for k, v in row.items()
            if v and k in ("offence", "cognizable", "bailable", "triable_by")
        ]
        if bits:
            lines.append(f"- Offence table: {'; '.join(bits)}")

    n_judg = payload.get("judgments_count")
    if n_judg is None:
        # A snapshot fetched before the storage policy dropped judgment records.
        n_judg = len(payload.get("judgments") or [])
    lines.append(f"- Reported judgments citing it: {n_judg}")
    if notes:
        lines.append(f"- Amended: yes ({len(notes)} chars of amendment history)")
    lines.append(f"- Source: {payload.get('url')}")
    lines += ["", operative, ""]
    if notes:
        lines += ["Amendment history:", "", textwrap.fill(notes, 88), ""]

    meta = {
        "provision": ref,
        "act": act["id"],
        "act_short": label,
        "act_title": act.get("short_title"),
        "act_year": act.get("act_year"),
        "role": "current" if in_force else "repealed",
        "in_force": str(bool(in_force)),
        "number": number,
        "heading": heading,
        "url": payload.get("url"),
        "n_judgments": n_judg,
        "words": section.get("words"),
        "superseded_by": (
            f"{labels.get(counterpart['act'], counterpart['act'])} "
            f"s.{counterpart['section']}"
            if counterpart and not in_force else ""
        ),
        "supersession_tentative": str(bool(tentative)),
        "replaces": (
            f"{labels.get(counterpart['act'], counterpart['act'])} "
            f"s.{counterpart['section']}"
            if counterpart and in_force else ""
        ),
    }
    return "\n".join(lines), meta


def render_rule(label: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """CPC Order/Rule responses are flat, not nested like section responses."""
    order_roman = payload.get("order") or ""
    order_arabic = payload.get("order_arabic")
    rule = str(payload.get("rule"))
    heading = clean(payload.get("heading")) or f"Rule {rule}"
    ref = f"{label} O.{order_arabic or order_roman} r.{rule}"
    operative, notes = split_notes(clean(payload.get("text")))

    lines = [f"## {ref} - {heading}", ""]
    lines.append(f"- Provision: {ref}")
    lines.append(f"- Order {order_roman}: {clean(payload.get('order_title'))}")
    lines.append(f"- Heading: {heading}")
    lines.append("- In force: yes")
    cites = payload.get("citations") or []
    if cites:
        lines.append(f"- Cross-references: {len(cites)}")
    lines.append(f"- Source: {payload.get('url')}")
    lines += ["", operative, ""]
    if notes:
        lines += ["Amendment history:", "", textwrap.fill(notes, 88), ""]

    meta = {
        "provision": ref,
        "act": "cpc",
        "act_short": label,
        "act_title": payload.get("act_title"),
        "role": "current",
        "in_force": "True",
        "order": str(order_arabic or order_roman),
        "rule": rule,
        "heading": heading,
        "url": payload.get("url"),
        "words": payload.get("words"),
        "n_citations": len(cites),
        "superseded_by": "",
        "replaces": "",
    }
    return "\n".join(lines), meta


# --------------------------------------------------------------------------
def build(manifest: dict[str, Any], out_dir: Path | None = None) -> dict[str, Any]:
    supersession = supersession_index()
    labels = manifest.get("act_labels", {})
    # out_dir exists so the gate can rebuild the corpus into a temporary
    # directory and compare it against what is committed. Same code path, so
    # the comparison is meaningful rather than a re-implementation of it.
    target_dir = out_dir or CORPUS
    target_dir.mkdir(parents=True, exist_ok=True)

    # Recover the selected provision list by re-running the fetcher's resolver
    # in dry-run mode: one source of truth for "what is in this corpus".
    from src.fetch import build_plan  # noqa: PLC0415 -- avoids import cycle at module load

    plan = build_plan(manifest, manifest["source"]["base"], dry=True)
    by_act: dict[str, list[Any]] = {}
    for t in plan.targets:
        by_act.setdefault(t.act, []).append(t)

    docs: list[dict[str, Any]] = []

    for entry in manifest["acts"]:
        act_id = entry["id"]
        act = act_metadata(act_id)
        label = entry.get("label") or labels.get(act_id, act_id.upper())
        blocks, metas = [], []
        for t in by_act.get(act_id, []):
            payload = read_raw(t.relpath)
            body, meta = render_section(act, label, payload, supersession, labels)
            blocks.append(body)
            metas.append(meta)

        header = [
            f"# {act.get('short_title')}",
            "",
            f"- Act id: `{act_id}`",
            f"- Citation label: **{label}**",
            f"- Act number: {act.get('act_number')} of {act.get('act_year')}",
            f"- Enacted: {act.get('enact_date')}",
            f"- In force: {'yes' if act.get('in_force') else 'NO - repealed'}",
            f"- Enforcement date: {act.get('enforcement_date')}",
            f"- Ministry: {act.get('ministry')}",
            f"- Role in this corpus: **{entry.get('role')}** - {entry.get('why')}",
            f"- Provisions included: {len(metas)} "
            f"of {act.get('section_count')} upstream",
            f"- Source: {act.get('url')}",
            "",
        ]
        path = target_dir / f"{act_id}.md"
        path.write_text("\n".join(header) + "\n" + "\n".join(blocks), encoding="utf-8")
        docs.append({
            "doc_id": act_id,
            "path": f"data/corpus/{act_id}.md",
            "label": label,
            "short_title": act.get("short_title"),
            "act_year": act.get("act_year"),
            "act_number": act.get("act_number"),
            "in_force": bool(act.get("in_force")),
            "role": entry.get("role"),
            "why": entry.get("why"),
            "n_provisions": len(metas),
            "provisions": [m["provision"] for m in metas],
        })

    # CPC rules: their own document.
    rule_targets = [t for t in plan.targets if t.kind == "rule"]
    if rule_targets:
        label = labels.get("cpc", "CPC")
        blocks, metas = [], []
        for t in rule_targets:
            body, meta = render_rule(label, read_raw(t.relpath))
            blocks.append(body)
            metas.append(meta)
        header = [
            "# Code of Civil Procedure, 1908 - selected Orders and Rules",
            "",
            "- Act id: `cpc`",
            f"- Citation label: **{label}**",
            "- Selected provisions: Orders and Rules, not sections",
            f"- Rules included: {len(metas)}",
            "- Source: https://indiacode.ecourtsindia.com/cpc/",
            "",
        ]
        path = target_dir / "cpc.md"
        path.write_text("\n".join(header) + "\n" + "\n".join(blocks), encoding="utf-8")
        docs.append({
            "doc_id": "cpc",
            "path": "data/corpus/cpc.md",
            "label": label,
            "short_title": "The Code of Civil Procedure, 1908",
            "act_year": 1908,
            "act_number": 5,
            "in_force": True,
            "role": "current",
            "why": "Court procedure as a document type distinct from a penal section.",
            "n_provisions": len(metas),
            "provisions": [m["provision"] for m in metas],
        })

    index = {
        "corpus_version": manifest.get("corpus_version"),
        "n_docs": len(docs),
        "n_provisions": sum(d["n_provisions"] for d in docs),
        "n_repealed_provisions": sum(
            d["n_provisions"] for d in docs if not d["in_force"]
        ),
        "docs": docs,
    }
    (target_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", action="store_true",
                    help="print corpus statistics without rebuilding")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    index_path = CORPUS / "index.json"

    if args.stats:
        if not index_path.exists():
            print("no corpus yet; run `python -m src.build`", flush=True)
            return 1
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = build(manifest)

    print(f"corpus_version {index['corpus_version']}")
    print(f"{index['n_docs']} documents, {index['n_provisions']} provisions "
          f"({index['n_repealed_provisions']} of them repealed)\n")
    print(f"{'doc':<26}{'label':<26}{'role':<10}{'in force':<10}provisions")
    for d in sorted(index["docs"], key=lambda x: (not x["in_force"], x["doc_id"])):
        print(f"{d['doc_id']:<26}{d['label']:<26}{d['role']:<10}"
              f"{'yes' if d['in_force'] else 'no':<10}{d['n_provisions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())