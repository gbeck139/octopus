# PyInstaller Distribution Setup - Summary

## What Was Implemented

This repository now includes a complete PyInstaller setup for bundling the radial non-planar slicer Python scripts and PrusaSlicer into a distributable package.

### Files Added

1. **requirements.txt** - Lists all Python dependencies needed for the project
2. **radial_slicer.spec** - PyInstaller specification file that controls the bundling process
3. **build.sh** - Linux/macOS build automation script
4. **build.bat** - Windows build automation script
5. **BUILD_INSTRUCTIONS.md** - Comprehensive guide for building the distribution
6. **USER_GUIDE.md** - Guide for end users on how to use the distributed package
7. **.github/workflows/build-release.yml** - GitHub Actions workflow for automated builds
8. **.gitignore** - Updated to exclude build artifacts

### Code Changes

**radial_non_planar_slicer/main.py**:
- Added `find_prusa_slicer()` function that looks for PrusaSlicer in multiple locations:
  - Bundled with the distribution (dist/PrusaSlicer/)
  - System PATH
  - Original absolute path as fallback
- Added `BASE_DIR` detection for both frozen (bundled) and script execution modes
- Improved portability across different systems

## How It Works

### Build Process

1. **Dependencies**: Install all Python libraries listed in requirements.txt
2. **PyInstaller**: Bundles Python interpreter, libraries, and application code into a single executable
3. **Data Files**: Includes configuration files and creates directory structure
4. **PrusaSlicer**: Reserved space for PrusaSlicer executable (must be added separately)

### Distribution Structure

```
dist/
├── radial_slicer (executable)
├── PrusaSlicer/
│   └── PrusaSlicer.AppImage (or prusa-slicer.exe)
└── radial_non_planar_slicer/
    ├── prusa_slicer/
    │   └── my_printer_config.ini
    ├── input_models/
    ├── output_models/
    ├── input_gcode/
    └── output_gcode/
```

## Benefits

### For Developers
- **Automated Builds**: GitHub Actions workflow builds for all platforms
- **Version Control**: Easy to track and manage releases
- **Reproducible**: Same build process across all environments

### For End Users
- **No Python Installation**: Everything bundled, just extract and run
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Simple**: Single executable with clear directory structure

## Usage

### Building Locally

**Linux/macOS:**
```bash
pip install -r requirements.txt
./build.sh
# Add PrusaSlicer to dist/PrusaSlicer/
```

**Windows:**
```batch
pip install -r requirements.txt
build.bat
REM Add PrusaSlicer to dist\PrusaSlicer\
```

### Automated Builds (CI/CD)

The GitHub Actions workflow automatically builds for all platforms when you:
- Push a version tag (e.g., `v1.0.0`)
- Manually trigger the workflow

### Distribution

After building:
1. Add PrusaSlicer to the dist/PrusaSlicer/ folder
2. Create an archive: `tar -czf radial_slicer.tar.gz dist/`
3. Share the archive with users
4. Users extract and run the executable directly

## Technical Details

### PyInstaller Configuration

The `radial_slicer.spec` file configures:
- **Entry Point**: `radial_non_planar_slicer/main.py`
- **Data Files**: Printer config, input models directory
- **Hidden Imports**: Libraries that PyInstaller might miss (numpy, pyvista, scipy, etc.)
- **One-File Mode**: Everything bundled into a single executable
- **Console Mode**: Runs as a console application (shows output)

### Dependencies Bundled

- **Core Libraries**: numpy, scipy, matplotlib
- **3D Processing**: pyvista (includes VTK)
- **Mesh Operations**: networkx
- **G-code Parsing**: pygcode
- **Python Runtime**: Python 3.8+ interpreter

### Size Considerations

The bundled application will be approximately:
- **Linux**: ~250-400 MB
- **Windows**: ~300-500 MB
- **macOS**: ~300-450 MB

This is normal for scientific Python applications with VTK/PyVista.

## PrusaSlicer Integration

### Why PrusaSlicer is Separate

PrusaSlicer is not bundled directly because:
1. **Licensing**: PrusaSlicer is AGPL-3.0, requires careful distribution
2. **Size**: PrusaSlicer is large (~200 MB)
3. **Platform-specific**: Different executables for each OS
4. **Updates**: Users may want specific PrusaSlicer versions

### How It's Found

The application looks for PrusaSlicer in this order:
1. `dist/PrusaSlicer/PrusaSlicer.AppImage` (Linux)
2. `dist/PrusaSlicer/prusa-slicer.exe` (Windows)
3. `dist/PrusaSlicer/prusa-slicer` (macOS)
4. System PATH (if installed globally)
5. Fallback to original hardcoded path

## Troubleshooting

### Build Issues

**"Module not found" during build**:
- Add the missing module to `hiddenimports` in the spec file
- Verify all dependencies are in requirements.txt

**Large executable size**:
- This is expected with scientific Python libraries
- Consider using UPX compression (already enabled)
- Split into multiple executables if needed

### Runtime Issues

**"PrusaSlicer not found" error**:
- Ensure PrusaSlicer is in the PrusaSlicer/ folder
- Check that the executable is actually executable (chmod +x on Linux)
- Verify the filename matches what the code expects

**Missing libraries on Linux**:
- Install system dependencies: `sudo apt-get install libgl1-mesa-glx libglib2.0-0`

## Future Enhancements

### Possible Improvements

1. **Command-line Arguments**: Add CLI options to specify model name, parameters
2. **GUI Wrapper**: Create a simple GUI for non-technical users
3. **Smaller Distribution**: Optimize dependencies, remove unused libraries
4. **Model Validation**: Check STL files before processing
5. **Progress Indicators**: Show build/slice progress
6. **Automatic PrusaSlicer Download**: Script to fetch PrusaSlicer automatically

### Advanced Features

1. **Plugins System**: Allow custom deformation functions
2. **Batch Processing**: Process multiple models at once
3. **Parameter Presets**: Save and load common configurations
4. **Cloud Integration**: Upload/download models from cloud storage

## License Considerations

When distributing:
- **Your Code**: Use your chosen license
- **Python Libraries**: Most are permissive (MIT, BSD)
- **PrusaSlicer**: AGPL-3.0 (requires source code availability if modified)
- **PyInstaller**: GPL with exception for bundled applications

Ensure you comply with all licenses in your distribution.

## Support and Documentation

- **BUILD_INSTRUCTIONS.md**: Detailed build guide
- **USER_GUIDE.md**: End-user documentation
- **This File**: Technical overview and summary
- **Repository Issues**: For bug reports and feature requests

## Conclusion

The repository is now set up for easy distribution using PyInstaller. Users can download a self-contained package without needing Python installed, while developers have automated build processes through GitHub Actions. The setup handles all the complexity of bundling scientific Python libraries while maintaining flexibility for different PrusaSlicer installations.
