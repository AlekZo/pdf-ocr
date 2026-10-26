@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set "PYEXE=%USERPROFILE%\.pdf-ocr\venv\Scripts\python.exe"
if not exist "%PYEXE%" (
  echo The environment is not installed yet - run setup.bat first.
  pause & exit /b 1
)
if "%~1"=="" (
  echo Drag one or more PDF files ^(or a folder^) onto ocr.bat.
  echo Command line:  ocr.bat file.pdf [--lang ru] [--dpi 250] [--structure] [--no-pdf] [--json] [--pages 1-3]
  pause & exit /b 0
)
"%PYEXE%" "%~dp0pdf_ocr.py" %*
set RC=%ERRORLEVEL%
echo.
if not "%RC%"=="0" echo Finished with errors ^(code %RC%^).
pause
exit /b %RC%
