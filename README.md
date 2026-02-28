# WordCraft

WordCraft is a tools-first writing assistant.

## UI structure

- `/` -> **Tools Home** (primary value proposition)
  - One-Word Substitution
  - Smart Match (multi-constraint)
  - Synonyms/Antonyms
  - Rhymes/Homonyms
- `/editor` -> **Writing Workspace** (deep mode)
  - Draft / Polish / Transform
  - Suggestion sidebar
  - Optional tools drawer

## Core APIs

Public:
- `POST /suggest`
- `POST /lexical`
- `POST /constraints`
- `POST /oneword`
- `POST /feedback` (ratings, optional auth)

Protected:
- Auth (`/auth/*`)
- Documents (`/documents*`)
- Saved words (`/saved-words*`)

## NLP Pipeline

WordCraft uses a unified NLP pipeline for Draft/Polish/Transform and Tools:

1. Intent detection
2. Candidate generation (context vocab + WordNet + derivations + optional ConceptNet)
3. POS/constraint filtering
4. Weighted ranking (semantic + context + grammar + frequency)
5. Explainable output (`score` + reason text)

Notes:
- `/suggest` returns ranked suggestions with `word`, `score`, `pos`, and `note` (reason).
- `/lexical` now returns both `results` and `details` (scored, reasoned entries).
- `/constraints` and `/oneword` return scored, explainable candidates.

## Local run

See:
- `QUICKSTART.md` for a fast setup
- `SETUP.md` for full setup and environment details

## Automated tests

- Runtime + regression NLP tests live in `backend/tests/`
- Run:
  - `python -m pytest backend/tests`

## Supervised Dataset + Reranker

WordCraft now includes a dataset + training scaffold for end-to-end quality improvement across editor modes and tools.

- Schema: `backend/ml/data/dataset_schema.json`
- Seed examples: `backend/ml/data/seed_gold.jsonl`
- Full guide: `backend/ml/DATASET_AND_TRAINING.md`

Commands:

- Build labeled ranker data:
  - `python -m backend.ml.scripts.build_dataset`
- Train reranker:
  - `python -m backend.ml.scripts.train_reranker`
- Evaluate reranker:
  - `python -m backend.ml.scripts.eval_reranker`
- Export and merge user rating feedback from MongoDB:
  - `python -m backend.ml.scripts.export_feedback_dataset --append-default`

Implicit positive feedback:
- Saving a word/rewrite to favorites auto-logs a positive event in `feedback_ratings`.
- Inserting/accepting editor suggestions also logs a positive event.
- These events are merged into training/eval datasets via `export_feedback_dataset`.
