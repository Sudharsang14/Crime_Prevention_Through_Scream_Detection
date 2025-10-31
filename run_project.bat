
@echo off
setlocal ENABLEDELAYEDEXPANSION
echo ===============================
echo  Scream Detection - One Click
echo ===============================

REM Change to script directory
cd /d "%~dp0"

REM Create venv if missing
if not exist venv (
  echo Creating virtual environment...
  py -3 -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

REM Extract features if missing
if not exist scream_features.csv (
  echo Extracting MFCC features to scream_features.csv ...
  python scripts\extract_features.py
)

REM Train models if missing
if not exist models\svm_model.pkl (
  echo Training models...
  python scripts\train_models.py
)

echo Starting web app...
python app.py

echo.
echo Press any key to close.
pause >nul
endlocal
