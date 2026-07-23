@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  AIFmri launcher - double-click me, or run me from anywhere.
REM
REM  cd's to its OWN folder first, so it can never serve an older
REM  copy that happens to be the current directory.
REM
REM  It picks an interpreter by actually importing the app rather
REM  than guessing a dependency list: an earlier version probed
REM  for "fastapi, onnxruntime", passed a Python that was missing
REM  onnx (a DIFFERENT package), and died on startup. The only
REM  honest test of "can this Python run AIFmri" is to try it.
REM ============================================================
cd /d "%~dp0"

echo.
echo   AIFmri  -  serving from: %CD%
echo.

if not exist "app\main.py" (
  echo   ERROR: app\main.py is not next to this script.
  echo   Extract the whole zip, then run START.bat from inside it.
  echo.
  pause
  exit /b 1
)

REM A project venv always wins if it can run the app.
set "PY="
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "import app.main" >nul 2>&1
  if !errorlevel! equ 0 set "PY=.venv\Scripts\python.exe"
)

REM Otherwise try each installed interpreter, best wheel support first.
if not defined PY (
  for %%V in (3.12 3.13 3.11 3.14 3.10) do (
    if not defined PY (
      py -%%V -c "import app.main" >nul 2>&1
      if !errorlevel! equ 0 set "PY=py -%%V"
    )
  )
)
if not defined PY (
  python -c "import app.main" >nul 2>&1
  if !errorlevel! equ 0 set "PY=python"
)

if not defined PY (
  echo   No Python on this machine can import AIFmri yet.
  echo   What each installed interpreter is missing:
  echo.
  for %%V in (3.12 3.13 3.11 3.14 3.10) do (
    py -%%V -c "import sys;print('   Python %%V  ->  '+sys.version.split()[0])" 2>nul >nul && (
      py -%%V -c "import sys;print('   Python %%V ('+sys.version.split()[0]+')')" 2>nul
      py -%%V -c "import app.main" 2>&1 | findstr /c:"ModuleNotFoundError"
    )
  )
  echo.
  echo   Install the dependencies into ONE of them, for example:
  echo.
  echo       py -3.12 -m pip install -r requirements.txt
  echo.
  echo   NOTE: onnx and onnxruntime are SEPARATE packages. You need both.
  echo.
  pause
  exit /b 1
)

REM Port 8000 is often reserved by Hyper-V/WSL on Windows; 8001 is safer.
set PORT=8001
echo   Interpreter: %PY%
echo   Open:        http://127.0.0.1:%PORT%
echo.
echo   Confirm the badge next to the AIFMRI title reads v0.19.0
echo.
%PY% -m uvicorn app.main:app --port %PORT%
echo.
echo   Server stopped.
pause
