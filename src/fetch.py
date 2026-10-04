"""Fetch a pinned statutory snapshot from the IndiaCode Statute API.

    python -m src.fetch --dry-run     # show the plan, touch nothing
    python -m src.fetch               # fetch into data/raw/, write the lockfile
    python -m src.fetch --check       # offline: verify checksums against lockfile

Why fetch at all, when the API is one HTTP call away?

Three reasons, in order of importance:

1. **Reproducibility.** An evaluation whose corpus can change underneath it
   measures nothing. `data/raw/FETCHED.json` pins the upstream build id, the
   upstream corpus date, and a SHA-256 per response. Two runs a year apart
   are then comparable, or provably not.
2. **Offline.** Everything downstream — chunking, indexing, retrieval metrics,
   the regression gate — must run with `AIP_OFFLINE=1` and no network. The
   raw snapshot is what makes that true.
3. **Rate-limit courtesy.** The API is keyless and public. Hundreds of
   requests are fine if they are sequential and identified; they are not fine
   if they look like a scraper.

Only legislative text is stored. The upstream corpus is "published with
commentary and other original matter under s.52(1)(q)(ii), Copyright Act
1957": the text of an Act, rule or order is not copyrightable, but the
editorial commentary layered on top of it is. This module therefore keeps
`section.text` and drops the HTML rendering, the judgments, and any note
field. If you extend it, keep that line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "manifest.json"
RAW = REPO / "data" / "raw"
LOCKFILE = RAW / "FETCHED.json"

# Identify yourself. The API is keyless and read-only; being a good citizen
# costs one header.
USER_AGENT = "legal-rag-corpus/1.0 (educational RAG project; contact: local)"

DEFAULT_BASE = "https://indiacode.ecourtsindia.com/api/v1"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class FetchError(RuntimeError):
    pass


def get_json(url: str, *, timeout: int = 45, retries: int = 4) -> dict[str, Any]:
    """GET with exponential backoff. Raises FetchError on giving up."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                json.JSONDecodeError) as exc:
            last = exc
            # 4xx other than 429 will not fix themselves; stop early.
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise FetchError(f"{url} -> HTTP {exc.code}") from exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise FetchError(f"{url} -> {type(last).__name__}: {last}") from last


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip third-party editorial matter from a payload before it is stored.

    IndiaCode's own licence line reads: "Published with commentary and other
    original matter under s.52(1)(q)(ii), Copyright Act 1957." The statutory
    text is a legislative work and nobody owns it, but the section payloads also
    carry `judgments`: 2732 records, and 53% of the bytes in `data/raw/`. Each
    record holds an `applied_to_this_section` field, which the API's own
    `source_note` describes as "the ratio decidendi as extracted from the order
    by eCourts India". That is a summary written by eCourts, not the judgment
    text and not the statute -- precisely the "original matter" the licence
    reserves.

    Committing it would mean redistributing material whose copyright is not ours
    to grant. The corpus does not need it: `build.py` only ever rendered
    `len(judgments)`, so the count is retained as `judgments_count` and the
    records are dropped.

    Not a legal opinion, and not a substitute for one. It is the conservative
    choice: keep what the pipeline uses, do not keep what we cannot license.
    See LICENSING.md.
    """
    if "judgments" in payload:
        records = payload.get("judgments") or []
        payload = dict(payload)
        payload["judgments_count"] = len(records)
        payload.pop("judgments", None)
    return payload


def write_raw(relpath: str, payload: dict[str, Any], *, force: bool,
              keep_judgments: bool = False) -> tuple[str, bool]:
    """Write one API response under data/raw/. Returns (sha256, was_written)."""
    path = RAW / relpath
    blob = json.dumps(
        payload if keep_judgments else sanitise_payload(payload),
        ensure_ascii=False, indent=1, sort_keys=True,
    )
    data = blob.encode("utf-8")
    if path.exists() and not force:
        return sha256_bytes(path.read_bytes()), False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data), True


def sanitise_existing() -> int:
    """Rewrite stored payloads in place, dropping judgments. No network.

    Lets an already-fetched snapshot be brought under the current storage policy
    offline, which matters because the checksums change when the policy does.
    Counts are preserved, so the built corpus is unaffected.
    """
    lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))
    changed = dropped = 0
    for relpath in lock["sha256"]:
        path = RAW / relpath
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "judgments" not in payload:
            continue
        dropped += len(payload.get("judgments") or [])
        path.write_bytes(json.dumps(
            sanitise_payload(payload), ensure_ascii=False, indent=1, sort_keys=True
        ).encode("utf-8"))
        changed += 1

    lock["sha256"] = {
        rel: sha256_bytes((RAW / rel).read_bytes())
        for rel in sorted(lock["sha256"])
        if (RAW / rel).exists()
    }
    lock["sanitised"] = True
    lock["sanitised_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    LOCKFILE.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"sanitised {changed} payload(s), dropped {dropped} judgment record(s); "
          f"lockfile checksums refreshed")
    return 0


# --------------------------------------------------------------------------
# Correspondence maps
# --------------------------------------------------------------------------
# The upstream API ships official old-generation -> new-generation
# correspondence for the 2024 recodification:
#
#     ipc-bns    Indian Penal Code 1860        -> Bharatiya Nyaya Sanhita 2023
#     crpc-bnss  Code of Criminal Procedure    -> Bharatiya Nagarik Suraksha Sanhita
#     iea-bsa    Indian Evidence Act 1872      -> Bharatiya Sakshya Adhiniyam
#
# Each row carries a `relation` (near-identical | close | weak | none) and a
# confidence `score`. This is the corpus's reason for existing: it gives a
# principled, upstream-authored definition of "this provision and that one say
# the same thing", which is exactly the distinction a retriever gets wrong.
STRONG_RELATIONS = ("near-identical", "close")


def fetch_mapping_pair(pair: str, base: str, *, force: bool = False) -> list[dict[str, Any]]:
    """Fetch one correspondence map, following pagination. Caches to disk."""
    path = RAW / f"mappings/{pair}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))["rows"]

    rows: list[dict[str, Any]] = []
    offset, total = 0, None
    while total is None or offset < total:
        page = get_json(f"{base}/mappings?pair={pair}&limit=500&offset={offset}")
        total = page["total"]
        rows.extend(page["mappings"])
        offset += page["count"]
        if not page["count"]:
            break

    write_raw(f"mappings/{pair}.json",
              {"pair": pair, "total": total, "count": len(rows), "rows": rows},
              force=force)
    return rows


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
@dataclass
class Target:
    """One thing to fetch: a provision of an act, or a CPC rule."""

    kind: str  # "section" | "article" | "rule"
    act: str
    number: str
    relpath: str
    url: str

    @property
    def label(self) -> str:
        if self.kind == "rule":
            return f"CPC O{self.act[-1] if self.act.startswith('o') else self.act} r.{self.number}"
        return f"{self.act} {self.kind} {self.number}"


@dataclass
class Plan:
    acts: list[dict[str, Any]] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_requests(self) -> int:
        return len(self.acts) + len(self.targets)


def mapped_sections(
    act_id: str,
    pair: str,
    side: str,
    relations: tuple[str, ...],
    min_score: float,
) -> list[str]:
    """Sections of `act_id` that `pair` maps strongly onto the other generation.

    `side` is "from" (the repealed provision, e.g. IPC s.302) or "to" (its
    successor, e.g. BNS s.103). Taking both sides of the same map is what puts
    near-duplicate text in one index, which is the retrieval trap this corpus
    is built around.
    """
    rows = fetch_mapping_pair(pair, DEFAULT_BASE)
    rels = set(relations)
    out: list[str] = []
    for r in rows:
        if r.get("relation") not in rels or r.get("score", 0) < min_score:
            continue
        # Both endpoints must exist. A row with a null counterpart is not a
        # supersession pair -- it is a provision the recodification dropped --
        # and including it on one side only would unbalance the two acts.
        if not (r.get("from_section") and r.get("to_section")):
            continue
        if side == "from" and r.get("from_act") == act_id:
            out.append(r["from_section"])
        elif side == "to" and r.get("to_act") == act_id:
            out.append(r["to_section"])
    return out


def resolve_sections(entry: dict[str, Any], act_meta: dict[str, Any]) -> list[str]:
    """Turn a manifest section spec into an explicit list of section numbers.

    Supported forms, in `data/manifest.json`:

        "sections": {"mode": "all"}
        "sections": {"mode": "explicit", "numbers": ["1", "103"]}
        "sections": {"mode": "headings", "contains": ["punishment for murder"]}
        "sections": {"mode": "mapped", "pair": "ipc-bns", "side": "to"}
        "sections": {"mode": "mapped", "pair": "ipc-bns", "side": "from",
                     "relations": ["near-identical"], "min_score": 0.9}

    plus an optional `"intersect_headings"` list applied after the mode
    resolves, so a map can be narrowed topically, and an optional
    `"also_include"` list unioned back in afterwards.

    `mapped` is the important one. It takes the correspondence map the API
    publishes for the 2024 recodification and selects the provisions upstream
    itself records as near-duplicates of the other generation. Curating the
    corpus this way means the old/new pairs are selected by an authoritative
    source rather than by my guesswork about which sections matter.

    `headings` exists for the Acts with no repealed counterpart, where topical
    selection is the only sensible criterion.
    """
    spec = entry.get("sections")
    if spec is None:
        raise ValueError(f"act {entry.get('id')!r} has no 'sections' spec")

    available = act_meta.get("sections") or []
    mode = spec["mode"]

    if mode == "all":
        chosen = [s["number"] for s in available]
    elif mode == "explicit":
        chosen = [str(n) for n in spec["numbers"]]
    elif mode == "headings":
        patterns = [re.compile(p, re.IGNORECASE) for p in spec["contains"]]
        chosen = [
            s["number"] for s in available
            if any(p.search(s.get("heading") or "") for p in patterns)
        ]
    elif mode == "mapped":
        chosen = mapped_sections(
            entry["id"],
            spec["pair"],
            spec.get("side", "to"),
            tuple(spec.get("relations", STRONG_RELATIONS)),
            float(spec.get("min_score", 0.0)),
        )
    else:
        raise ValueError(f"unknown sections mode {mode!r}")

    headings = {s["number"]: s.get("heading") or "" for s in available}
    words = {s["number"]: int(s.get("words") or 0) for s in available}

    # Narrow a resolved set topically. Intersecting rather than unioning is
    # deliberate: it is a filter, not an extension.
    keep = spec.get("intersect_headings")
    if keep:
        pats = [re.compile(p, re.IGNORECASE) for p in keep]
        chosen = [n for n in chosen if any(p.search(headings.get(n, "")) for p in pats)]

    known = set(headings)
    missing = [n for n in chosen if n not in known]
    if missing:
        raise ValueError(
            f"act {entry['id']!r}: sections not present upstream: {missing}"
        )

    min_words = spec.get("min_words")
    if min_words:
        chosen = [n for n in chosen if words.get(n, 0) >= min_words]

    # `also_include` is the escape hatch from a threshold. The map filters are
    # deliberately strict -- IPC s.323 is `near-identical` to BNS s.115 but
    # scores 0.88, and IPC s.341 and s.348 are only `close` at 0.82 and 0.75 --
    # so three genuine old/new pairs are excluded by a cut-off set for corpus
    # size. Loosening the cut-off would admit hundreds of near-misses to fix
    # three; naming the three is auditable and the reason is recorded here.
    for number in spec.get("also_include", ()):
        chosen.append(str(number))

    # Stable order, no duplicates. Section numbers are strings that mix
    # integers ("103") and alphanumerics ("124A"), so sort naturally-ish.
    seen: set[str] = set()
    out = [n for n in dict.fromkeys(chosen) if not (n in seen or seen.add(n))]
    if not out:
        raise ValueError(f"act {entry['id']!r} resolved to zero sections")
    return out


def build_plan(manifest: dict[str, Any], base: str, *, dry: bool) -> Plan:
    """Resolve the whole manifest into a concrete list of requests."""
    plan = Plan()
    act_metas: dict[str, dict[str, Any]] = {}

    for entry in manifest["acts"]:
        act_id = entry["id"]
        meta_url = f"{base}/acts/{act_id}"
        meta_path = f"acts/{act_id}.json"
        cached = RAW / meta_path
        if cached.exists():
            act_metas[act_id] = json.loads(cached.read_text(encoding="utf-8"))
        elif dry:
            # In dry-run we still need the section list to resolve selectors,
            # so fetch metadata (cheap, one request per act) and discard.
            act_metas[act_id] = get_json(meta_url)
        else:
            plan.acts.append({"id": act_id, "url": meta_url, "relpath": meta_path})
            act_metas[act_id] = None  # populated on the next pass

    # Two passes: any act whose metadata was not cached needs fetching before
    # its section list can be resolved.
    unresolved = [a for a in manifest["acts"] if act_metas.get(a["id"]) is None]
    for entry in unresolved:
        act_id = entry["id"]
        act_metas[act_id] = get_json(f"{base}/acts/{act_id}")

    for entry in manifest["acts"]:
        act_id = entry["id"]
        meta = act_metas[act_id]
        # `unit`, `in_force`, `short_title` live under the nested "act" object;
        # `sections` is top level.
        unit = meta.get("act", {}).get("unit", "section")
        numbers = resolve_sections(entry, meta)
        plan.warnings.append(
            f"act {act_id}: upstream has {len(meta.get('sections') or [])} "
            f"{unit}s, manifest selects {len(numbers)}"
        )
        for n in numbers:
            kind = "article" if unit == "article" else "section"
            plan.targets.append(
                Target(
                    kind=kind,
                    act=act_id,
                    number=n,
                    relpath=f"{unit}s/{act_id}__{n}.json",
                    url=f"{base}/{act_id}/{kind}/{n}",
                )
            )

    for entry in manifest.get("rules", []):
        order, rule = str(entry["order"]), str(entry["rule"])
        plan.targets.append(
            Target(
                kind="rule",
                act=entry.get("act", "cpc"),
                number=rule,
                relpath=f"rules/{entry.get('act', 'cpc')}__o{order}__r{rule}.json",
                url=f"{base}/cpc/order/{order}/rule/{rule}",
            )
        )

    # Duplicate provision refs across acts would silently overwrite each other.
    dupes = [t.relpath for t in plan.targets
             if [x.relpath for x in plan.targets].count(t.relpath) > 1]
    if dupes:
        raise ValueError(f"duplicate provisions in manifest: {sorted(set(dupes))}")

    return plan


# --------------------------------------------------------------------------
# Lockfile
# --------------------------------------------------------------------------
def build_lockfile(plan: Plan, base: str, entries: dict[str, str], meta: dict[str, Any]) -> dict[str, Any]:
    """The provenance record. Commit this alongside data/raw/."""
    by_kind: dict[str, int] = {}
    for t in plan.targets:
        by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
    return {
        "corpus_version": meta.get("corpus_version"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream": {
            "base": base,
            "build": meta.get("build"),
            "corpus_date": meta.get("corpus_date"),
            "source_of_record": meta.get("source_of_record"),
            "licence": meta.get("licence"),
        },
        "counts": {
            "files": len(entries),
            "provisions": len(plan.targets),
            "by_kind": by_kind,
        },
        "sha256": dict(sorted(entries.items())),
    }


def check_offline() -> int:
    """Verify every file on disk still matches the lockfile. Exit code = mismatches."""
    if not LOCKFILE.exists():
        print(f"missing {LOCKFILE}; run `python -m src.fetch` first", file=sys.stderr)
        return 1
    lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))
    bad = 0
    for relpath, digest in lock["sha256"].items():
        path = RAW / relpath
        if not path.exists():
            print(f"MISSING  {relpath}")
            bad += 1
        elif sha256_bytes(path.read_bytes()) != digest:
            print(f"CHANGED  {relpath}")
            bad += 1
    total = len(lock["sha256"])
    print(f"{total - bad}/{total} files match {LOCKFILE.name}")
    return 1 if bad else 0


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the manifest and print the plan; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="refetch files that already exist on disk")
    ap.add_argument("--check", action="store_true",
                    help="offline: verify data/raw against the lockfile and exit")
    ap.add_argument("--sanitise", action="store_true",
                    help=("offline: drop stored judgment records under the current "
                          "storage policy and refresh the lockfile"))
    ap.add_argument("--keep-judgments", action="store_true",
                    help=("store the full payloads including eCourts judgment "
                          "summaries. Off by default; see LICENSING.md"))
    ap.add_argument("--delay", type=float, default=0.25,
                    help="seconds between requests (courtesy pause)")
    args = ap.parse_args(argv)

    if args.check:
        return check_offline()
    if args.sanitise:
        return sanitise_existing()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = build_plan(manifest, args.base, dry=args.dry_run)

    for w in plan.warnings:
        print(f"  note: {w}")
    print(f"plan: {len(plan.acts)} act-metadata + {len(plan.targets)} provisions")

    if args.dry_run:
        kinds: dict[str, int] = {}
        for t in plan.targets:
            kinds[t.act] = kinds.get(t.act, 0) + 1
        for act, n in sorted(kinds.items()):
            print(f"  {act:<28} {n:>4} provisions")
        print(f"  {'TOTAL':<28} {len(plan.targets):>4}")
        return 0

    entries: dict[str, str] = {}

    # Act metadata first: the lockfile records it and it documents provenance.
    meta_all: dict[str, Any] = {}
    for item in plan.acts:
        payload = get_json(item["url"])
        meta_all.update({k: v for k, v in payload.get("act", {}).items()
                         if k in ("build", "corpus_date")})
        digest, written = write_raw(item["relpath"], payload, force=args.force)
        entries[item["relpath"]] = digest
        print(f"  [{'+' if written else '='}] {item['relpath']}")

    fetched = 0
    failures: list[tuple[Target, str]] = []
    for i, t in enumerate(plan.targets, start=1):
        if (RAW / t.relpath).exists() and not args.force:
            entries[t.relpath] = sha256_bytes((RAW / t.relpath).read_bytes())
            continue
        # One bad manifest entry must not throw away an hour of fetching.
        # Record it, keep going, and report at the end with a nonzero exit.
        try:
            payload = get_json(t.url)
        except FetchError as exc:
            failures.append((t, str(exc)))
            continue
        digest, _ = write_raw(t.relpath, payload, force=args.force,
                              keep_judgments=args.keep_judgments)
        entries[t.relpath] = digest
        fetched += 1
        if fetched % 50 == 0:
            print(f"  ... {i}/{len(plan.targets)}")
        time.sleep(args.delay)

    if failures:
        print(f"\n  {len(failures)} provision(s) could not be fetched:", file=sys.stderr)
        for t, err in failures:
            print(f"    {t.label:<24} {t.url}\n      {err}", file=sys.stderr)
        print("  Fix data/manifest.json and re-run; completed files are kept.",
              file=sys.stderr)

    # Pull /meta so the lockfile records the exact upstream build we pinned.
    try:
        m = get_json(f"{args.base}/meta")
        meta_all = {
            "build": m.get("build"),
            "corpus_date": m.get("corpus_date"),
            "source_of_record": (m.get("currency") or {}).get("source"),
            "licence": m.get("licence"),
        }
        write_raw("meta.json", m, force=args.force)
        entries["meta.json"] = sha256_bytes((RAW / "meta.json").read_bytes())
    except FetchError as exc:
        print(f"  warning: could not fetch /meta: {exc}", file=sys.stderr)

    # Fold in the files that were already on disk and so never went through the
    # fetch loop: act metadata and the correspondence maps. The lockfile is the
    # reproducibility record, so it must cover every input the corpus was built
    # from -- not just the provisions.
    for entry in manifest["acts"]:
        rel = f"acts/{entry['id']}.json"
        if (RAW / rel).exists():
            entries[rel] = sha256_bytes((RAW / rel).read_bytes())
    for pair in manifest.get("mapping_pairs", []):
        rel = f"mappings/{pair}.json"
        if (RAW / rel).exists():
            entries[rel] = sha256_bytes((RAW / rel).read_bytes())

    lock = build_lockfile(plan, args.base, entries,
                          {**meta_all, "corpus_version": manifest.get("corpus_version")})
    if failures:
        lock["failed"] = [{"target": t.relpath, "url": t.url, "error": e}
                          for t, e in failures]
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {LOCKFILE.relative_to(REPO)}: "
          f"{len(entries)} files, "
          f"build {lock['upstream']['build']}, corpus {lock['upstream']['corpus_date']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())