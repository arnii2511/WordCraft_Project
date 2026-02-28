# WordCraft Setup & Run Guide

Use this guide to run WordCraft on any new computer.

## 1) Prerequisites

- Python `3.11+`
- Node.js `20+` (recommended `22 LTS`)
- MongoDB `7+` (local install or Docker)
- Git (optional, for cloning)

---

## 2) Get the project

```bash
git clone <your-repo-url>
cd wordcraft
```

Or copy the project folder manually and open terminal in project root.

---

## 3) Backend setup (FastAPI + NLP)

### Windows (cmd)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

> Note: model download can take 1-2 minutes the first time.

---

## 4) Backend environment file

Create `backend/.env` using `backend/.env.example`:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=wordcraft
JWT_SECRET=replace-with-a-long-random-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080
CORS_ORIGINS=http://localhost:5173
```

---

## 5) Start MongoDB

### Option A: Local MongoDB service

Start MongoDB so it listens on `localhost:27017`.

#### If you use MongoDB Compass

1. Open MongoDB Compass.
2. Click `New Connection`.
3. Use connection string:
   - `mongodb://localhost:27017`
4. Click `Connect`.
5. You do **not** need to manually create collections.
   - `users`, `documents`, and `saved_words` will be created automatically on first write.
6. Optional: create database `wordcraft` manually in Compass for visibility.

### Option B: Docker

```bash
docker run -d --name wordcraft-mongo -p 27017:27017 mongo:7
```

If using Docker with Compass, connect to:
- `mongodb://localhost:27017`

---

## 5.1) If you use MongoDB Atlas (optional)

1. In Atlas, create a cluster and database user.
2. Allow your IP in `Network Access`.
3. Copy Atlas URI and put it in `backend/.env`:

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/...
```

---

## 6) Run backend

From project root (with virtualenv active):

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Check:

- Health: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

---

## 7) Frontend setup

Open a second terminal in project root:

```bash
npm install
npm run dev
```

Frontend URL:

- `http://localhost:5173`

UI routes:

- `/` -> Tools-first home
- `/editor` -> writing workspace (deep mode)

Optional frontend env file in root (`.env`):

```env
VITE_API_URL=http://localhost:8000
```

---

## 8) Daily startup steps

1. Start MongoDB
2. Activate virtualenv
3. Run backend (`uvicorn`)
4. Run frontend (`npm run dev`)

---

## 9) Feature access rules (current)

### Public (no login required)

- `POST /suggest`
- `POST /lexical`
- `POST /constraints`
- `POST /oneword`
- `POST /feedback` (ratings for suggestions/tools/rewrites)

### Protected (JWT required)

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /documents`
- `GET /documents`
- `GET /documents/{id}`
- `PUT /documents/{id}`
- `DELETE /documents/{id}`
- `POST /saved-words`
- `GET /saved-words`
- `DELETE /saved-words/{id}`

---

## 10) NLP engine behavior (current)

Unified pipeline used by Suggestions + Tools:

1. Intent detection (selection -> blank -> sentence)
2. Candidate generation (context vocab + WordNet + derivational forms + optional ConceptNet)
3. POS / constraint filtering
4. Weighted ranking (semantic + context + grammar + frequency)
5. Explainable output (`score` + reason string)

Mode behavior:
- Draft: context-aware suggestion with grammar fit
- Polish: clarity + grammar-safe rewrites
- Transform: stronger style shift on button trigger, meaning-preserving

Tool behavior:
- Lexical (`/lexical`): returns `results` and scored `details`
- Smart Match (`/constraints`): strict or best-effort ranked constraint matches
- One-Word (`/oneword`): phrase-to-word ranked substitutions

---

## 11) Quick validation checklist

### Guest mode

- Suggestions and tools work
- Smart Match works
- One-Word substitution works
- Save is blocked

### Logged-in mode

- Register/login works
- Document save/load/delete works
- Saved words add/list/delete works

### Session history

- Snapshot behavior is session-only (`sessionStorage`)
- Clears when browser session ends

---

## 12) Integration checks (recommended)

Backend compile check:

```bash
python -m compileall backend
```

Frontend build check:

```bash
npm run build
```

Optional lint check:

```bash
npm run lint
```

Automated NLP/runtime + regression test suite:

```bash
python -m pytest backend/tests
```

---

## 13) Supervised dataset + reranker training (optional but recommended)

This repo now includes a unified dataset/training pipeline for editor + tool quality.

Files:
- `backend/ml/data/dataset_schema.json`
- `backend/ml/data/seed_gold.jsonl`
- `backend/ml/DATASET_AND_TRAINING.md`

Commands:

```bash
python -m backend.ml.scripts.build_dataset
python -m backend.ml.scripts.train_reranker
python -m backend.ml.scripts.eval_reranker
python -m backend.ml.scripts.export_feedback_dataset --append-default
```

The builder creates labeled candidate rows for:
- Draft/Polish/Transform suggestion tasks
- Lexical tools
- Smart Match constraints
- One-word substitution

Recommended loop:
- Build a baseline dataset once and keep it as your eval baseline.
- Collect user ratings in MongoDB Atlas (`feedback_ratings`).
- Favorite/save and editor insert/accept actions are automatically captured as positive feedback.
- Export/merge feedback rows only when developers decide to retrain.
