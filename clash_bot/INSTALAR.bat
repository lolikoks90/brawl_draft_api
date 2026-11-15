@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalar bot de Clash Royale
echo ==============================================
echo   Instalando el bot de Clash Royale
echo ==============================================
echo.
echo El bot necesita Python 3.12 (las versiones mas nuevas aun no son compatibles
echo con todas las librerias). Puede convivir con otras versiones de Python.
echo.
set "PY="
for %%V in (3.12 3.13 3.11) do (
    if not defined PY ( py -%%V --version >nul 2>&1 && set "PY=py -%%V" )
    if not defined PY ( py -V:%%V --version >nul 2>&1 && set "PY=py -V:%%V" )
)
if not defined PY (
    py --version >nul 2>&1
    if not errorlevel 1 (
        echo Instalando Python 3.12 con el instalador de Python...
        echo Si te hace alguna pregunta, responde que si ^(Y^).
        py install 3.12
    )
)
set "PY="
for %%V in (3.12 3.13 3.11) do (
    if not defined PY ( py -%%V --version >nul 2>&1 && set "PY=py -%%V" )
    if not defined PY ( py -V:%%V --version >nul 2>&1 && set "PY=py -V:%%V" )
)
if not defined PY (
    echo.
    echo No he podido instalar Python 3.12 automaticamente.
    echo.
    echo 1. Se va a abrir la pagina de descargas de Python para Windows.
    echo 2. Busca "Python 3.12" y descarga "Windows installer ^(64-bit^)".
    echo 3. Al instalar, marca la casilla "Add python.exe to PATH".
    echo 4. Cuando termine, vuelve a hacer doble clic en INSTALAR.bat
    echo.
    start "" https://www.python.org/downloads/windows/
    pause
    exit /b 1
)
echo Usando:
%PY% --version
echo.
echo Instalando librerias (puede tardar 10-15 minutos, no cierres esta ventana)...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Ha habido un error. Haz una captura de esta ventana y pideme ayuda.
    pause
    exit /b 1
)
echo.
echo ==============================================
echo   Listo. Ahora haz doble clic en INICIAR.bat
echo ==============================================
pause
