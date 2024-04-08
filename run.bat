chcp 65001
@echo off
:: 检测管理员权限
NET SESSION >nul 2>&1
if %errorLevel% == 0 (
    echo 管理员权限 - YES
) else (
    echo 管理员权限 - NO
    echo 请用管理员权限执行！
    pause
    exit /b
)

:: 判断RailwayTurnoutGuard是否在运行
tasklist /FI "WINDOWTITLE eq RailwayTurnoutGuard" 2>NUL
if "%ERRORLEVEL%"=="0" (
    echo RailwayTurnoutGuard 正在运行，2 秒后强制中止进程...
    timeout /t 2 >nul
    taskkill /F /FI "WINDOWTITLE eq RailwayTurnoutGuard" >nul 2>&1
    echo RailwayTurnoutGuard killed.
)

echo 2 秒后启动 RailwayTurnoutGuard ...
timeout /t 2 >nul
echo Starting server.py...
cd /d %~dp0
start "RailwayTurnoutGuard" python server.py

timeout /t 5 >nul
