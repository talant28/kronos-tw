@echo off
cd /d "%~dp0Kronos"

set MASTER_ADDR=localhost
set MASTER_PORT=29500
set WORLD_SIZE=1
set RANK=0
set LOCAL_RANK=0
set PYTHONPATH=%~dp0Kronos;%PYTHONPATH%

REM Always copy latest config before training
copy /Y "%~dp0config_tw.py" "%~dp0Kronos\finetune\config_tw.py" >nul
copy /Y "%~dp0config_tw.py" "%~dp0Kronos\finetune\config.py" >nul

echo ============================================================
echo  GPU Check
echo ============================================================
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')"

echo.
echo ============================================================
echo  Step 1: Fine-tune Tokenizer (auto-resumes if already done)
echo ============================================================
time /t
python finetune\train_tokenizer.py
if %ERRORLEVEL% neq 0 (
    echo FAILED: Tokenizer training error
    echo Shutdown cancelled due to error.
    pause
    exit /b 1
)
echo Tokenizer done.
time /t

echo.
echo ============================================================
echo  Step 2: Fine-tune Predictor
echo ============================================================
time /t
python finetune\train_predictor.py
if %ERRORLEVEL% neq 0 (
    echo FAILED: Predictor training error
    echo Shutdown cancelled due to error.
    pause
    exit /b 1
)
echo Predictor done.
time /t

echo.
echo ============================================================
echo  All done. Checkpoints saved to: %~dp0checkpoints\
echo  Shutting down in 60 seconds... Press Ctrl+C to cancel.
echo ============================================================
shutdown /s /t 60 /c "Kronos training complete. Shutting down."
pause
