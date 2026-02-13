# Radial Slicer - User Guide

## For End Users (Using the Distributed Package)

### Installation

1. Download the `radial_slicer` package (ZIP or TAR.GZ) for your platform
2. Extract the archive to a location of your choice
3. The package is ready to use - no Python installation required!

### First Run

**Linux/macOS:**
```bash
cd dist
./radial_slicer
```

**Windows:**
```batch
cd dist
radial_slicer.exe
```

### Directory Structure

The distribution includes these folders:

- `PrusaSlicer/` - Contains the bundled PrusaSlicer executable
- `radial_non_planar_slicer/input_models/` - Place your STL files here
- `radial_non_planar_slicer/output_models/` - Deformed STL files are saved here
- `radial_non_planar_slicer/input_gcode/` - Intermediate gcode files
- `radial_non_planar_slicer/output_gcode/` - Final radial gcode files
- `radial_non_planar_slicer/prusa_slicer/` - PrusaSlicer configuration

### Basic Workflow

1. **Prepare your model:**
   - Save your STL file in `radial_non_planar_slicer/input_models/`
   - Name it according to the MODEL_NAME in the script (default: `3DBenchy.stl`)

2. **Run the slicer:**
   - Execute the `radial_slicer` application
   - The process will:
     - Load your STL
     - Apply radial deformation
     - Call PrusaSlicer for planar slicing
     - Transform the gcode back to radial coordinates

3. **Find your output:**
   - Radial gcode: `radial_non_planar_slicer/output_gcode/{MODEL_NAME}_reformed.gcode`
   - Deformed STL: `radial_non_planar_slicer/output_models/{MODEL_NAME}_deformed.stl`

### Customizing Parameters

To change the model being processed or deformation parameters, you'll need to:

1. Edit the main.py file before building, OR
2. Use the source code version with Python installed

### System Requirements

- **Operating System:** Linux, macOS, or Windows (64-bit)
- **RAM:** 4GB minimum (8GB recommended for large models)
- **Storage:** ~500MB for the application + space for your models
- **Display:** Required for mesh visualization

### Troubleshooting

**Error: "PrusaSlicer not found"**
- Ensure PrusaSlicer is in the `PrusaSlicer/` folder
- On Linux, make sure the AppImage is executable: `chmod +x PrusaSlicer/PrusaSlicer.AppImage`

**Application doesn't start**
- Check that you've extracted the entire archive, not just the executable
- On Linux, you may need to install system libraries (mesa, libX11)

**Large models cause crashes**
- The application subdivides meshes to ~1 million points
- Try reducing your input model complexity
- Increase available RAM

### Support

For issues and questions:
- Check the BUILD_INSTRUCTIONS.md for advanced configuration
- Review the source repository: https://github.com/gbeck139/octopus
- Based on: https://github.com/jyjblrd/Radial_Non_Planar_Slicer

## For Developers (Building from Source)

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for complete build instructions.

Quick start:
```bash
pip install -r requirements.txt
./build.sh  # or build.bat on Windows
```
