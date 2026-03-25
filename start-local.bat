@echo off
setlocal
cd /d %~dp0

start "WordCraft Core API" cmd /k "venv\Scripts\python -m uvicorn backend.main_core:app --reload --host 127.0.0.1 --port 8000"
start "WordCraft Lexical API" cmd /k "venv\Scripts\python -m uvicorn backend.main_lexical:app --reload --host 127.0.0.1 --port 8001"
start "WordCraft Editor API" cmd /k "venv\Scripts\python -m uvicorn backend.main_editor:app --reload --host 127.0.0.1 --port 8002"
start "WordCraft Frontend" cmd /k "npm run dev"

endlocal
