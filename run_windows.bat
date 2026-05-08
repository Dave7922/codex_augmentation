@echo off
setlocal

cd /d "%~dp0"

echo PCB Defect Augmentation - Windows launcher
echo.

set "ENV_NAME=pcb311"

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda was not found.
    echo Install Anaconda or Miniconda, then reopen Command Prompt and run this file again.
    echo Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

conda run -n %ENV_NAME% python --version >nul 2>nul
if errorlevel 1 (
    echo Creating conda environment: %ENV_NAME%
    conda create -y -n %ENV_NAME% python=3.11
    if errorlevel 1 (
        echo Failed to create conda environment %ENV_NAME%.
        echo.
        pause
        exit /b 1
    )
)

echo Using conda environment: %ENV_NAME%
conda run -n %ENV_NAME% python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    echo.
    pause
    exit /b 1
)

conda run -n %ENV_NAME% python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install project dependencies.
    echo.
    pause
    exit /b 1
)

echo.
echo Starting application...
conda run -n %ENV_NAME% python app.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    echo.
    pause
    exit /b 1
)

endlocal
