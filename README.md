radial_non_planar_slicer based on https://github.com/jyjblrd/Radial_Non_Planar_Slicer?tab=readme-ov-file

## Building and Distribution

To create a distributable package with PyInstaller that bundles the radial slicer Python scripts and PrusaSlicer, see [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md).

Quick start:
```bash
./build.sh  # Linux/macOS
build.bat   # Windows
```

This will create a `dist` folder with everything users need (after you add PrusaSlicer).
