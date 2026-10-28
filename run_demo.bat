@echo off
REM GeoVision demo launcher: starts the API + web app and opens it in the browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Follow the setup steps in README.md first.
    pause
    exit /b 1
)
start "GeoVision server (close this window to stop)" cmd /k ".venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000"
echo Starting GeoVision... (loading the model takes a few seconds)
timeout /t 10 /nobreak >nul
start "" http://127.0.0.1:8000
echo.
echo GeoVision is running:
echo   Web app  : http://127.0.0.1:8000
echo   API docs : http://127.0.0.1:8000/docs
echo Close the "GeoVision server" window to stop it.
