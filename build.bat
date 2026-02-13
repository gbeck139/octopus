@echo off
REM Build script for creating distributable radial slicer package on Windows

echo =========================================
echo Radial Slicer Build Script (Windows)
echo =========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Install dependencies
echo.
echo Step 1: Installing Python dependencies...
pip install -r requirements.txt

REM Run PyInstaller
echo.
echo Step 2: Building executable with PyInstaller...
pyinstaller radial_slicer.spec --clean

REM Create PrusaSlicer directory in dist
echo.
echo Step 3: Setting up PrusaSlicer in distribution...
if not exist "dist\PrusaSlicer" mkdir "dist\PrusaSlicer"

REM Instructions for PrusaSlicer
echo.
echo =========================================
echo PrusaSlicer Setup Instructions
echo =========================================
echo.
echo To complete the distribution, you need to add PrusaSlicer:
echo.
echo 1. Download PrusaSlicer from https://www.prusa3d.com/page/prusaslicer_424/
echo 2. Install PrusaSlicer
echo 3. Copy prusa-slicer.exe to: dist\PrusaSlicer\prusa-slicer.exe
echo 4. Copy all DLL dependencies from PrusaSlicer installation to dist\PrusaSlicer\
echo.

REM Create output directories
echo.
echo Step 4: Creating output directories...
if not exist "dist\radial_non_planar_slicer\input_gcode" mkdir "dist\radial_non_planar_slicer\input_gcode"
if not exist "dist\radial_non_planar_slicer\output_gcode" mkdir "dist\radial_non_planar_slicer\output_gcode"
if not exist "dist\radial_non_planar_slicer\output_models" mkdir "dist\radial_non_planar_slicer\output_models"

REM Copy config files
echo.
echo Step 5: Copying configuration files...
xcopy /E /I /Y "radial_non_planar_slicer\prusa_slicer" "dist\radial_non_planar_slicer\prusa_slicer"
if exist "radial_non_planar_slicer\input_models" (
    xcopy /E /I /Y "radial_non_planar_slicer\input_models" "dist\radial_non_planar_slicer\input_models"
) else (
    echo Warning: No input models found to copy
)

echo.
echo =========================================
echo Build Complete!
echo =========================================
echo.
echo Your distributable package is in the 'dist' folder.
echo Don't forget to add PrusaSlicer as described above.
echo.
echo To distribute:
echo   1. Add PrusaSlicer to dist\PrusaSlicer\
echo   2. Zip the entire 'dist' folder
echo   3. Users can extract and run 'radial_slicer.exe' directly
echo.
pause
