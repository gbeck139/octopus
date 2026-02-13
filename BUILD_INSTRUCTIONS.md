# Building and Distributing the Radial Slicer

This guide explains how to bundle the radial slicer Python scripts and PrusaSlicer into a distributable package using PyInstaller.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (to clone the repository)

## Quick Start

### Linux/macOS

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the build script
./build.sh

# 3. Add PrusaSlicer to the distribution
# Download PrusaSlicer AppImage from https://www.prusa3d.com/page/prusaslicer_424/
# Copy it to: dist/PrusaSlicer/PrusaSlicer.AppImage
# Make it executable:
chmod +x dist/PrusaSlicer/PrusaSlicer.AppImage

# 4. Test the build
cd dist
./radial_slicer

# 5. Package for distribution
cd ..
tar -czf radial_slicer_linux.tar.gz dist/
```

### Windows

```batch
REM 1. Install dependencies
pip install -r requirements.txt

REM 2. Run the build script
build.bat

REM 3. Add PrusaSlicer to the distribution
REM Download and install PrusaSlicer from https://www.prusa3d.com/page/prusaslicer_424/
REM Copy prusa-slicer.exe and DLLs to: dist\PrusaSlicer\

REM 4. Test the build
cd dist
radial_slicer.exe

REM 5. Package for distribution
REM Create a ZIP archive of the dist folder
```

## What Gets Bundled

The build process creates a self-contained distribution that includes:

1. **Python Executable**: The radial slicer compiled into a standalone executable
2. **Python Dependencies**: All required Python libraries (numpy, pyvista, pygcode, etc.)
3. **Configuration Files**: Printer configuration INI file
4. **Input/Output Directories**: Pre-created directories for models and gcode
5. **PrusaSlicer**: Space reserved for the PrusaSlicer executable (you must add this)

## Directory Structure

After building, your `dist` folder will have this structure:

```
dist/
├── radial_slicer (or radial_slicer.exe on Windows)
├── PrusaSlicer/
│   └── PrusaSlicer.AppImage (or prusa-slicer.exe on Windows)
└── radial_non_planar_slicer/
    ├── prusa_slicer/
    │   └── my_printer_config.ini
    ├── input_models/
    ├── output_models/
    ├── input_gcode/
    └── output_gcode/
```

## How PrusaSlicer is Found

The application will look for PrusaSlicer in the following order:

1. `dist/PrusaSlicer/PrusaSlicer.AppImage` (Linux)
2. `dist/PrusaSlicer/prusa-slicer.exe` (Windows)
3. `dist/PrusaSlicer/prusa-slicer` (macOS/Linux)
4. System PATH (if PrusaSlicer is installed system-wide)

## Manual Build (Advanced)

If you prefer to build manually:

```bash
# Install dependencies
pip install -r requirements.txt

# Run PyInstaller with the spec file
pyinstaller radial_slicer.spec --clean

# Create necessary directories
mkdir -p dist/PrusaSlicer
mkdir -p dist/radial_non_planar_slicer/{input_gcode,output_gcode,output_models}

# Copy configuration files
cp -r radial_non_planar_slicer/prusa_slicer dist/radial_non_planar_slicer/
cp -r radial_non_planar_slicer/input_models dist/radial_non_planar_slicer/

# Add PrusaSlicer
# (Download and copy as described above)
```

## Customizing the Build

### Modifying the PyInstaller Spec File

The `radial_slicer.spec` file controls how PyInstaller bundles the application. You can customize:

- **Icon**: Add `icon='path/to/icon.ico'` to the EXE section
- **Additional Data Files**: Add more entries to the `datas` list
- **Hidden Imports**: Add any missing imports to the `hiddenimports` list

### Including Additional Models

To include sample models in the distribution:

1. Place your STL files in `radial_non_planar_slicer/input_models/`
2. Rebuild using the build script

The models will automatically be included in the distribution.

## Distribution

### Creating a Distributable Package

**Linux/macOS:**
```bash
tar -czf radial_slicer_$(uname -s)_$(uname -m).tar.gz dist/
```

**Windows:**
Create a ZIP archive of the `dist` folder using Windows Explorer or:
```batch
powershell Compress-Archive -Path dist -DestinationPath radial_slicer_windows.zip
```

### Sharing with Users

Users can download your package and:

1. Extract the archive
2. Navigate to the extracted `dist` folder
3. Run the `radial_slicer` executable directly (no Python installation needed)

## Troubleshooting

### "PrusaSlicer not found" Error

**Solution**: Make sure PrusaSlicer is in the `dist/PrusaSlicer/` folder and is executable.

**Linux:**
```bash
chmod +x dist/PrusaSlicer/PrusaSlicer.AppImage
```

### Missing Dependencies

**Solution**: If the bundled app is missing Python libraries, add them to `hiddenimports` in the spec file:

```python
hiddenimports = [
    'numpy',
    'missing_library_name',
]
```

Then rebuild.

### Large Distribution Size

**Solution**: The distribution can be large (200-500 MB) due to:
- PyVista and VTK libraries
- NumPy and SciPy
- PrusaSlicer

This is normal for a bundled Python application with scientific libraries.

To reduce size:
- Remove unused dependencies from requirements.txt
- Use UPX compression (already enabled in the spec file)
- Consider splitting the distribution (Python app separate from PrusaSlicer)

## Development vs Production

- **Development**: Run `python radial_non_planar_slicer/main.py` directly
- **Production**: Use the bundled executable from `dist/radial_slicer`

## License

This build system is provided as-is. Ensure you comply with all licenses:
- Your code
- Python and its libraries
- PrusaSlicer (AGPL-3.0)

## Support

For issues with:
- **Build process**: Check this README and the build scripts
- **Python code**: Check the main repository documentation
- **PrusaSlicer**: Visit https://www.prusa3d.com/page/prusaslicer_424/
