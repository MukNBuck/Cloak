@echo off
REM Shortcut for the Windows 5070 box.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_crag_5070.ps1"
pause
