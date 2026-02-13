# Packaging Instructions

To package the `radial_non_planar_slicer` as a standalone executable:

1.  **Install Dependencies**:
    Make sure you have all required packages installed.
    ```bash
    pip install pyinstaller pyvista numpy scipy networkx pygcode
    ```
    (Note: If `pygcode` is not available on PyPI, you might need to install it from source or include the `pygcode` folder if it's a local library. If it is a local folder in your workspace, it will be bundled automatically).

2.  **Run PyInstaller**:
    Run the following command from the `radial_non_planar_slicer` directory:

    ```bash
    cd radial_non_planar_slicer
    pyinstaller --noconfirm --onedir --console --name "radial_slicer" --add-data "prusa_slicer;prusa_slicer" --add-data "input_models;input_models" --add-data "input_gcode;input_gcode" --add-data "output_models;output_models" --add-data "output_gcode;output_gcode"  main.py
    ```

3.  **Add Prusa Slicer**:
    After the build completes, go to `dist/radial_slicer/`.
    Copy your generic Prusa Slicer folder (containing `prusa-slicer-console.exe`) into this directory and rename it to `PrusaSlicer`.
    
    Structure should look like:
    ```
    dist/radial_slicer/
        radial_slicer.exe
        PrusaSlicer/
            prusa-slicer-console.exe
            ...
        utils.py
        ...
    ```

4.  **Run**:
    Double click `radial_slicer.exe`.
