# WordCraft Quickstart (2-Minute)

## 1) Prerequisites
- Python 3.11+
- Node.js 20+
- MongoDB running on `localhost:27017`

## 1.1) MongoDB Compass (if you use it)
1. Open Compass.
2. Create/open connection: `mongodb://localhost:27017`
3. Connect.
4. No manual collections needed (app creates them on first save).

## 2) Backend (Terminal 1)

### Windows (cmd)
```bat
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
copy backend\.env.example backend\.env
python -m uvicorn backend.main:app --reload --port 8000
```

### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
cp backend/.env.example backend/.env
python -m uvicorn backend.main:app --reload --port 8000
```

## 3) Frontend (Terminal 2)
```bash
npm install
npm run dev
```

Open: `http://localhost:5173`

App routes:
- `/` = Tools Home (One-Word, Smart Match, Synonyms/Antonyms, Rhymes/Homonyms)
- `/editor` = Writing workspace (Draft/Polish/Transform)

## 4) Verify
- Backend docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Quick API smoke checks:
- `POST /suggest` returns `suggestions[]` with `word`, `score`, `pos`, `note`
- `POST /lexical` returns `results[]` and `details[]`
- `POST /constraints` returns ranked `results[]` with `reason`
- `POST /oneword` returns ranked `results[]` with `reason`

## 5) Access model
- Public: suggestions + lexical + smart match
- Public: one-word substitution
- Login required: save docs + saved words

## 6) Run automated NLP checks
```bash
python -m pytest backend/tests
```
