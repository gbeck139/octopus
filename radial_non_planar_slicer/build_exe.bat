@echo off
echo Installing requirements...
pip install -r requirements.txt
pip install pyinstaller

echo Packaging application...
pyinstaller --clean --noconfirm --onedir --console --name "radial_slicer" ^
    --add-data "prusa_slicer;prusa_slicer" ^
    --add-data "input_models;input_models" ^
    --add-data "input_gcode;input_gcode" ^
    --add-data "output_models;output_models" ^
    --add-data "output_gcode;output_gcode" ^
    main.py

echo.
echo ==========================================
echo Build complete! The executable is in:
echo district\radial_slicer\radial_slicer.exe
echo ==========================================
echo.
echo Please copy your 'PrusaSlicer' folder to:
echo dist\radial_slicer\PrusaSlicer
echo.
echo The PrusaSlicer folder MUST contain 'prusa-slicer-console.exe'.
echo.
pause
