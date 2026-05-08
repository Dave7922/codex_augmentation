@echo off
setlocal

cd /d "%~dp0"

echo PCB Defect Augmentation - Windows launcher
echo.

set "ENV_NAME=pcb311"
set "SAM_CHECKPOINT=checkpoints\sam_vit_b_01ec64.pth"
set "SAM_URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
set "PCTNET_WEIGHTS=third_party\PCT-Net-Image-Harmonization-main\pretrained_models\PCTNet_ViT.pth"
set "PCTNET_URL=https://github.com/rakutentech/PCT-Net-Image-Harmonization/raw/main/pretrained_models/PCTNet_ViT.pth"

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

echo Installing PyTorch CPU support for SAM...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo Failed to install PyTorch.
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

if not exist "checkpoints" mkdir "checkpoints"
if not exist "%SAM_CHECKPOINT%" (
    echo Downloading SAM pretrained checkpoint...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%SAM_URL%' -OutFile '%SAM_CHECKPOINT%'"
    if errorlevel 1 (
        echo Failed to download SAM checkpoint.
        echo Download it manually from:
        echo %SAM_URL%
        echo and save it as:
        echo %SAM_CHECKPOINT%
        echo.
        pause
        exit /b 1
    )
) else (
    echo SAM checkpoint already exists: %SAM_CHECKPOINT%
)

if not exist "third_party\PCT-Net-Image-Harmonization-main\pretrained_models" mkdir "third_party\PCT-Net-Image-Harmonization-main\pretrained_models"
if not exist "%PCTNET_WEIGHTS%" (
    echo Downloading PCT-Net pretrained weights...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%PCTNET_URL%' -OutFile '%PCTNET_WEIGHTS%'"
    if errorlevel 1 (
        echo Failed to download PCT-Net weights.
        echo Download it manually from:
        echo %PCTNET_URL%
        echo and save it as:
        echo %PCTNET_WEIGHTS%
        echo.
        pause
        exit /b 1
    )
) else (
    echo PCT-Net weights already exist: %PCTNET_WEIGHTS%
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
