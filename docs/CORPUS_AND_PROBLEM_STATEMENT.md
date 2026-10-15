# Legal-RAG corpus and problem statement

## Project problem statement

Legal-RAG is a retrieval-augmented question-answering system over a curated subset
of Indian statutes. Its central problem is **temporal legal correctness**: Indian
criminal law was recodified on 1 July 2024, and the repealed IPC, CrPC, and Indian
Evidence Act remain highly similar to their successors, the BNS, BNSS, and BSA.
A retriever can therefore return a persuasive, on-topic citation to a provision
that is no longer in force.

The project aims to make the answer's legal status part of the retrieval and
answering contract. It should:

- retrieve in-force provisions by default and keep historical provisions
  available for explicitly historical questions;
- avoid presenting repealed law as current, and identify a successor when a
  historical provision is used;
- cite only provisions actually retrieved, and refuse when the supplied sources
  do not support a safe answer;
- measure retrieval quality and enforce regression thresholds.

The offline gate measures retrieval and answer-contract behavior. It does **not**
establish comprehensive factual correctness or legal advice quality; those
dimensions require separate evaluation.

## Corpus snapshot and scope

The committed corpus is built from a pinned **IndiaCode Statute API** snapshot
(`data/manifest.json`, corpus version `2026-10-01`). It contains **930 selected
provisions across 15 Acts**. It is a curated subset, not a complete copy of each
Act. The selected old/new codes are both retained deliberately: the system needs
to encounter old text to recognize and filter it for current-law questions.

| Group | Act | Provisions included |
|---|---|---:|
| Current criminal code | Bharatiya Nyaya Sanhita (BNS), 2023 | 110 |
| Current criminal procedure | Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 | 96 |
| Current evidence code | Bharatiya Sakshya Adhiniyam (BSA), 2023 | 91 |
| Repealed criminal code | Indian Penal Code (IPC), 1860 | 118 |
| Repealed criminal procedure | Code of Criminal Procedure (CrPC), 1973 | 95 |
| Repealed evidence code | Indian Evidence Act (IEA), 1872 | 89 |
| Other legislation | Arbitration and Conciliation Act | 37 |
| Other legislation | Consumer Protection Act | 40 |
| Other legislation | Indian Contract Act | 42 |
| Other legislation | Code of Civil Procedure (CPC) | 27 |
| Other legislation | Limitation Act | 32 |
| Other legislation | Motor Vehicles Act | 26 |
| Other legislation | Information Technology Act | 54 |
| Other legislation | Negotiable Instruments Act | 46 |
| Other legislation | Specific Relief Act | 27 |
| **Total** | **15 Acts** | **930** |

Of the selected provisions, 302 are from the three repealed codes (IPC, CrPC,
and IEA); the other 628 are from current codes and other Acts. The searchable
index contains **996 chunks** over these 15 corpus documents, using
`local/sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions;
2,600-character chunk size). The index metadata reports 318 repealed chunks;
that count is chunk-level, not a count of unique repealed provisions.

The recodification links come from the upstream correspondence maps used by the
manifest. They are retrieval aids, not a complete legal cross-reference table.
Some links are explicitly marked weak. Check the source provision and the
commencement/status notes before relying on a link.

## What the corpus contains—and does not contain

Each corpus document contains selected statutory text with provision reference,
heading, status, and where available, repeal/successor metadata. It is intended
for statute retrieval, not general legal research across every source type.

- **Included:** selected statutory provisions from the 15 Acts listed above;
  both current and repealed criminal/evidence-code generations; status and
  supersession metadata used by the retrieval pipeline.
- **Not included:** every section of every listed Act, case law/judgments,
  general legal commentary, or a comprehensive collection of regulations and
  subordinate legislation.
- **Example coverage gap:** BNS section 4, which lists kinds of punishment, is
  not among the selected BNS provisions. Therefore the corpus cannot reliably
  answer “What are the kinds of punishment under BNS?” from that provision,
  even though it contains other BNS sections about specific offences and their
  penalties. BNSS is a procedural code; the general classification of
  punishments is a BNS matter, not a BNSS list.
- **Rights and provenance:** the source snapshot is attributed to India Code,
  republished by eCourts India. The build retains statutory text and excludes
  judgment records/editorial commentary; see `LICENSING.md` for the policy and
  `data/manifest.json` for selection details.

## Why one question may answer and another may be refused

“What is wrongful restraint?” maps to **BNS section 126**, which is included in
the corpus. “Explain the various types of punishments under BNSS?” has a source
and coverage mismatch: the general list is in BNS section 4, the prompt names
BNSS, and BNS section 4 is absent from this subset. Retrieval may find nearby
procedure provisions mentioning sentences or punishment, but those do not
support a complete list. The answer sufficiency check is intended to withhold
an unsupported response rather than fill the gap from model memory.

For an answerable query, ask about a provision present in the corpus. Expanding
coverage requires adding the relevant provision to the manifest/snapshot,
rebuilding `data/corpus/` and `data/index/`, and then evaluating the updated
retrieval behavior.

## Where to inspect the source material

- `data/manifest.json` — selected Acts and provision-selection rules.
- `data/corpus/*.md` — built, searchable statutory corpus (15 files).
- `data/index/index.json` — index dimensions, model, chunk count, and checksums.
- `CONCEPT_NOTE.md` — design rationale, retrieval measurements, and the core
  repealed-law failure case.
- `LICENSING.md` — source/licensing and storage-policy analysis.
