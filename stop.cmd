@echo off
REM Stop the assistant. Your documents, your index and your settings are all
REM kept - this only stops the program.

setlocal
cd /d "%~dp0"

echo Stopping your document assistant...
docker compose down
echo.
echo Stopped. Double-click start.cmd whenever you want it back.
echo.
pause
