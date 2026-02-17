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
