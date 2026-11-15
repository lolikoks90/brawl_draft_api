@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PY="
for %%V in (3.12 3.13 3.11) do (
    if not defined PY ( py -%%V --version >nul 2>&1 && set "PY=py -%%V" )
    if not defined PY ( py -V:%%V --version >nul 2>&1 && set "PY=py -V:%%V" )
)
if not defined PY (
    echo No encuentro Python 3.12. Primero haz doble clic en INSTALAR.bat
    pause
    exit /b 1
)
%PY% lanzador.py
if errorlevel 1 (
    echo.
    echo El lanzador se ha cerrado con un error. Haz una captura de esta ventana y pideme ayuda.
    pause
)
