@echo off
setlocal
cd /d %~dp0
set PYTHONPATH=src
python -m voice_recognition.web_frontend
endlocal
