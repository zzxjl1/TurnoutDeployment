@echo off
:: ������ԱȨ��
NET SESSION >nul 2>&1
if %errorLevel% == 0 (
    echo ����ԱȨ�� - YES
) else (
    echo ����ԱȨ�� - NO
    echo ���ù���ԱȨ��ִ�У�
    pause
    exit /b
)

:: �ж�RailwayTurnoutGuard�Ƿ�������
tasklist /FI "WINDOWTITLE eq RailwayTurnoutGuard" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo RailwayTurnoutGuard �������У�2 ���ǿ����ֹ����...
    timeout /t 2 >nul
    taskkill /F /FI "WINDOWTITLE eq RailwayTurnoutGuard" >nul 2>&1
    echo RailwayTurnoutGuard killed.
)

echo 2 ������� RailwayTurnoutGuard ...
timeout /t 2 >nul
echo Starting server.py...
start "RailwayTurnoutGuard" python server.py

timeout /t 5 >nul
