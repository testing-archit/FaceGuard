@echo off
rem Windows: double-click to start FaceGuard. First run installs components (needs internet once).
cd /d "%~dp0"
where py >nul 2>nul && (py -3 launcher.py) || (python launcher.py)
if errorlevel 1 pause
