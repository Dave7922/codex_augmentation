@echo off
setlocal

cd /d "%~dp0"

echo PCB Defect Augmentation - Windows launcher
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PYTHON_CMD=python"
    ) else (
        echo Python was not found.
        echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
        echo During installation, enable "Add python.exe to PATH".
        echo.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        echo.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate the virtual environment.
    echo.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    echo.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install project dependencies.
    echo.
    pause
    exit /b 1
)

echo.
echo Starting application...
python app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo.
    pause
    exit /b 1
)

endlocal
