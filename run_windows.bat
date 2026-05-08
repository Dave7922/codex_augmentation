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

if /I "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" (
    echo Conda environment %ENV_NAME% is already active.
) else (
    echo Checking conda environment: %ENV_NAME%
    conda env list | findstr /R /C:"^%ENV_NAME% " >nul 2>nul
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

    echo Activating conda environment: %ENV_NAME%
    call conda activate %ENV_NAME%
    if errorlevel 1 (
        echo Failed to activate conda environment %ENV_NAME%.
        echo If you installed Conda recently, reopen Anaconda Prompt or Command Prompt and try again.
        echo.
        pause
        exit /b 1
    )
)

python --version
if errorlevel 1 (
    echo Python is not available in conda environment %ENV_NAME%.
    echo Recreating the environment may fix this:
    echo   conda remove -n %ENV_NAME% --all
    echo   conda create -y -n %ENV_NAME% python=3.11
    echo.
    pause
    exit /b 1
)

echo Installing dependencies in conda environment: %ENV_NAME%
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
