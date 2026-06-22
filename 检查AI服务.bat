@echo off
setlocal
pushd "%~dp0"
call "deploy\simple_start\check_ai_service.bat"
popd
