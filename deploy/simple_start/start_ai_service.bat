@echo off
setlocal
cd /d "%~dp0..\.."

if not exist "logs" mkdir "logs"
if not exist "data\online" mkdir "data\online"
if not exist "models\tabpfn_cache" mkdir "models\tabpfn_cache"

set MODEL_DEVICE=cpu
set USE_BACKUP=true
set CPU_THREADS=4
set OMP_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4
set ONLINE_HISTORY_DIR=%CD%\data\online
set TABPFN_MODEL_CACHE_DIR=%CD%\models\tabpfn_cache
set LOG_LEVEL=INFO

echo ==========================================
echo Defluor AI dosing recommendation service
echo ==========================================
echo.
echo Keep this window open while the service is running.
echo API:
echo   http://127.0.0.1:8000/api/v1/dose/recommend/simple
echo.
echo Ready check:
echo   http://127.0.0.1:8000/api/v1/ready
echo.
echo Starting...
echo.
echo When you see "Uvicorn running on http://0.0.0.0:8000", the API is ready to be checked.
echo Do NOT close this window. Closing it stops the API service.
echo.

python -m uvicorn serve:app --host 0.0.0.0 --port 8000

echo.
echo Service stopped. Press any key to close this window.
pause >nul
