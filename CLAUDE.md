# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Octopus is a multi-strategy 4-axis (XYZC) printer slicing system for non-planar additive manufacturing. It targets CX-ZB kinematics printers (XYZ + rotational C axis) and provides multiple slicing strategies for handling overhangs through non-planar layer approaches.

## Architecture

### Core Pipeline Pattern

All slicers follow a three-stage pipeline: **Deform → Slice → Reform**
1. **Deform**: Transform 3D mesh into a modified geometry suitable for planar slicing
2. **Slice**: Use a standard slicer (PrusaSlicer) on the deformed mesh
3. **Reform**: Back-transform the resulting G-code into 4-axis (XYZC) coordinates using saved transform metadata (JSON)

### Major Components

- **gui/**: Qt6 C++ desktop application with tab-based UI (Prepare, Preview, Monitor). Uses profiles (printer, material, process) and a setup wizard for configuring paths to PrusaSlicer, Python, and printer config. Built with CMake.
- **native_slicer/**: Python `cxzb_slicer` package — a modular 4-axis slicing pipeline with inverse kinematics, singularity handling, Poisson-based vector fields, toolpath generation, and G-code output (Duet/Marlin). Has pytest tests.
- **radial_non_planar_slicer/**: Radial slicing with smooth blend between Cartesian (low Z) and radial (high Z) regions. Parameters: `angle_base`, `angle_factor`, `transition_z`, `blend_height`.
- **hybrid_slicer/**: WIP. Combines Cartesian and radial approaches using per-vertex rotation fields computed from surface overhang analysis with Laplacian smoothing.
- **conic_slicer/**: Conic deformation approach (`z' = z + cone_factor * r`) with mesh subdivision.
- **generic_non_planar_slicer/**: Tetrahedral mesh approach using tetgen, NetworkX, and scipy optimization.
- **firmware/**: Printer firmware binary and system configuration files.

### Slicer I/O Convention

Each slicer directory follows a standard layout:
- `input_models/` — STL input files
- `output_models/` — Deformed mesh output
- `input_gcode/` — Planar-sliced G-code
- `output_gcode/` — Back-transformed 4-axis G-code
- `prusa_slicer/` — PrusaSlicer configuration INI files

## Build Commands

### GUI (C++ / Qt6)

```bash
cd gui
cmake -B build
cmake --build build
```

Requires Qt6 (Widgets, 3DCore, 3DRender, 3DInput, 3DExtras, Quick). Post-build copies `slicerbundle/` to the executable directory.

### Python Slicers

Each Python slicer has its own virtual environment. Typical usage:

```bash
cd <slicer_dir>
source venv/bin/activate
python main.py [args]
```

### Native Slicer Tests

```bash
cd native_slicer
source venv/bin/activate
pytest                    # run all tests
pytest tests/test_foo.py  # run a single test file
```

## Key Python Dependencies

numpy, scipy, pyvista (mesh/visualization), networkx (graph algorithms), pygcode (G-code parsing), tetgen/open3d (3D geometry)

## Git Workflow

- Main branch: `main`
- Contributor branches: `contributor/<name>`
