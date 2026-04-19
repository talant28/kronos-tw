@echo off
REM =====================================================
REM  Kronos Fine-tune 執行腳本（Windows）
REM  硬體：RTX 4080 12GB, 64GB RAM, i9-13980HX
REM =====================================================

echo ============================================================
echo  Kronos 台股 Fine-tune 執行流程
echo ============================================================

REM 先確認在正確的目錄（你的工作資料夾）
set WORK_DIR=%~dp0
set KRONOS_DIR=%WORK_DIR%Kronos
cd /d "%WORK_DIR%"

echo 工作目錄：%WORK_DIR%
echo Kronos 目錄：%KRONOS_DIR%

REM ── 步驟 0：環境確認 ──────────────────────────────────────
echo.
echo [步驟 0] 確認 GPU 狀態...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo.

REM ── 步驟 1：Qlib 格式轉換 ────────────────────────────────
echo [步驟 1] 將 filtered CSV 轉換為 Qlib 格式...
echo 預估時間：5-10 分鐘
python convert_to_qlib.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Qlib 轉換失敗，請查看錯誤訊息
    pause
    exit /b 1
)
echo ✅ Qlib 轉換完成

REM ── 步驟 2：Clone Kronos 並安裝依賴 ─────────────────────
echo.
echo [步驟 2] 設定 Kronos 環境...
if not exist "%KRONOS_DIR%" (
    echo 正在 clone Kronos repo...
    git clone https://github.com/shiyu-coder/Kronos.git "%KRONOS_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo ❌ Clone 失敗，請手動執行：
        echo    git clone https://github.com/shiyu-coder/Kronos.git Kronos
        pause
        exit /b 1
    )
)
pip install -r "%KRONOS_DIR%\requirements.txt" -q
echo ✅ Kronos 環境設定完成

REM ── 步驟 3：複製 Taiwan config ───────────────────────────
echo.
echo [步驟 3] 複製 Taiwan 專屬 config...
copy /Y "%WORK_DIR%config_tw.py" "%KRONOS_DIR%\finetune\config_tw.py"
copy /Y "%WORK_DIR%config_tw.py" "%KRONOS_DIR%\finetune\config.py"
echo ✅ config_tw.py 複製到 Kronos/finetune/

REM ── 步驟 4：Qlib 資料前處理 ──────────────────────────────
echo.
echo [步驟 4] Kronos 資料前處理（產出 train/val/test pkl）...
echo 預估時間：5-15 分鐘
cd /d "%KRONOS_DIR%"
python finetune\qlib_data_preprocess.py
if %ERRORLEVEL% neq 0 (
    echo ⚠️  前處理出現警告（可能是正常的 Qlib 訊息），繼續執行...
)
echo ✅ 資料前處理完成

REM ── 步驟 5：Fine-tune Tokenizer ──────────────────────────
echo.
echo [步驟 5] Fine-tune Tokenizer（Kronos-base）
echo 預估時間：3-5 小時
echo 開始時間：
time /t
torchrun --standalone --nproc_per_node=1 finetune\train_tokenizer.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Tokenizer 訓練失敗
    pause
    exit /b 1
)
echo ✅ Tokenizer Fine-tune 完成
time /t

REM ── 步驟 6：Fine-tune Predictor ──────────────────────────
echo.
echo [步驟 6] Fine-tune Predictor（Kronos-base）
echo 預估時間：8-15 小時
echo 開始時間：
time /t
torchrun --standalone --nproc_per_node=1 finetune\train_predictor.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Predictor 訓練失敗
    pause
    exit /b 1
)
echo ✅ Predictor Fine-tune 完成
time /t

REM ── 步驟 7：回測驗證 ──────────────────────────────────────
echo.
echo [步驟 7] 回測驗證...
python finetune\qlib_test.py --device cuda:0
echo ✅ 回測完成

echo.
echo ============================================================
echo  全部完成！結果存放在：
echo  Checkpoints：%WORK_DIR%checkpoints\
echo  回測結果：  %WORK_DIR%backtest_results\
echo ============================================================
pause
