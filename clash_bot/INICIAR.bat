@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo No encuentro Python. Primero haz doble clic en INSTALAR.bat
    pause
    exit /b 1
)
%PY% lanzador.py
if errorlevel 1 (
    echo.
    echo El lanzador se ha cerrado con un error. Copia este texto y pideme ayuda.
    pause
)
