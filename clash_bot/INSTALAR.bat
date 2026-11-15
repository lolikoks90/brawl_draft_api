@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalar bot de Clash Royale
echo ==============================================
echo   Instalando el bot de Clash Royale
echo ==============================================
echo.
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo No encuentro Python en este ordenador.
    echo.
    echo 1. Se va a abrir la pagina de descarga de Python.
    echo 2. Descarga e instala Python.
    echo 3. MUY IMPORTANTE: marca la casilla "Add python.exe to PATH" al instalar.
    echo 4. Cuando termine, vuelve a hacer doble clic en INSTALAR.bat
    echo.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
)
echo Python encontrado:
%PY% --version
echo.
echo Instalando librerias (puede tardar varios minutos, no cierres esta ventana)...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Ha habido un error. Copia el texto de esta ventana y pideme ayuda.
    pause
    exit /b 1
)
echo.
echo ==============================================
echo   Listo. Ahora haz doble clic en INICIAR.bat
echo ==============================================
pause
