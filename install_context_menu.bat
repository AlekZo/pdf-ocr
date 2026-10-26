@echo off
rem Adds the right-click "OCR (PaddleOCR)" drop-down for PDFs and folders (current user only, no admin).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0context_menu.ps1" install
pause
