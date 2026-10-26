@echo off
setlocal
rem One-time setup: creates a private Python environment and installs PaddleOCR.
rem The environment lives in %USERPROFILE%\.pdf-ocr (NOT in this folder, so Nextcloud does not sync ~1 GB).
rem It is not under AppData on purpose: Microsoft Store Python redirects AppData to a private location.
rem   setup.bat        -> GPU build if an NVIDIA driver is found, otherwise CPU build
rem   setup.bat cpu    -> force the CPU build

set "VENV=%USERPROFILE%\.pdf-ocr\venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "PY="

rem PaddlePaddle 3.2 supports Python 3.9-3.13. Pick the newest one that is installed.
for %%V in (3.13 3.12 3.11 3.10 3.9) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  rem Fallback: python or python3 on PATH, for example Anaconda or a Python not registered with the py launcher
  for %%C in (python python3) do (
    if not defined PY (
      %%C -c "import sys; sys.exit(0 if (3,9)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul && set "PY=%%C"
    )
  )
)
if not defined PY (
  echo No suitable Python 3.9-3.13 was found. What this PC has:
  echo --- py launcher:
  py --list 2>nul || echo     ^(py launcher not found^)
  echo --- python on PATH:
  python --version 2>nul || echo     ^(none^)
  echo.
  echo Install Python 3.12 from https://www.python.org/downloads/ - it can live next to other versions.
  echo Tick "Add python.exe to PATH" and "py launcher" in the installer, then run setup.bat again.
  pause & exit /b 1
)
echo Using: %PY%

if not exist "%PYEXE%" (
  %PY% -m venv "%VENV%" || (echo Could not create the environment & pause & exit /b 1)
)
if not exist "%PYEXE%" (
  echo The environment was not created at %VENV%
  pause & exit /b 1
)
"%PYEXE%" -m pip install --upgrade pip

set "GPU=0"
if /i not "%~1"=="cpu" (
  where nvidia-smi >nul 2>nul && set "GPU=1"
)

set "DONE=0"
if "%GPU%"=="1" (
  echo NVIDIA driver found - installing the CUDA 12.9 GPU build of PaddlePaddle.
  "%PYEXE%" -m pip uninstall -y paddlepaddle >nul 2>nul
  "%PYEXE%" -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ && (
    "%PYEXE%" -m pip install paddleocr==3.7.0 pymupdf numpy && set "DONE=1"
  )
)
if "%DONE%"=="0" (
  "%PYEXE%" -m pip uninstall -y paddlepaddle-gpu >nul 2>nul
  "%PYEXE%" -m pip install -r "%~dp0requirements.txt" || (echo Installation failed & pause & exit /b 1)
)

rem Extra packages for --structure (tables and layout). Not fatal if it fails.
"%PYEXE%" -m pip install "paddlex[ocr]==3.7.2" || echo Could not install the --structure extras; plain OCR still works.

echo.
echo Done. Now drag a PDF ^(or a folder with PDFs^) onto ocr.bat, or run:  ocr.bat file.pdf
echo The OCR console shows "device: gpu:0" or "device: cpu" so you can see what is used.
pause
