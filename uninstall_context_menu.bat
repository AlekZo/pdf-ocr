@echo off
rem Removes the right-click "OCR (PaddleOCR)" drop-down.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0context_menu.ps1" uninstall
pause
