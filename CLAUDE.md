# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Octopus is a multi-strategy 4-axis (XYZC) printer slicing system for non-planar additive manufacturing. It targets CX-ZB kinematics printers (XYZ + rotational C axis) and provides multiple slicing strategies for handling overhangs through non-planar layer approaches.

## Architecture

### Core Pipeline Pattern

All slicers follow a three-stage pipeline: **Deform -> Slice -> Reform**
1. **Deform**: Transform 3D mesh into a modified geometry suitable for planar slicing
2. **Slice**: Use a standard slicer (PrusaSlicer) on the deformed mesh
3. **Reform**: Back-transform the resulting G-code into 4-axis (XYZC) coordinates using saved transform metadata (JSON)

### Major Components

- **gui/**: Qt6 C++ desktop application with tab-based UI (Prepare, Preview, Monitor). Uses profiles (printer, material, process) and a setup wizard for configuring paths to PrusaSlicer, Python, and printer config. Built with CMake.
- **native_slicer/**: Python `cxzb_slicer` package -- a modular 4-axis slicing pipeline with inverse kinematics, singularity handling, Poisson-based vector fields, toolpath generation, and G-code output (Duet/Marlin). Has pytest tests in `native_slicer/tests/` (22 tests).
- **radial_non_planar_slicer/**: Radial slicing with smooth blend between Cartesian (low Z) and radial (high Z) regions. Parameters: `angle_base`, `angle_factor`, `transition_z`, `blend_height`.
- **hybrid_slicer/**: WIP. Combines Cartesian and radial approaches using per-vertex rotation fields computed from surface overhang analysis with Laplacian smoothing.
- **conic_slicer/**: Conic deformation approach (`z' = z + cone_factor * r`) with mesh subdivision and wave modulation (sine, sawtooth, curvature, azimuthal, normal).
- **generic_non_planar_slicer/**: Tetrahedral mesh approach using tetgen, NetworkX, and scipy optimization. Single monolithic main.py.
- **firmware/**: Printer firmware binary and system configuration files (Duet 3).

### Slicer I/O Convention

Each slicer directory follows a standard layout:
- `input_models/` -- STL input files
- `output_models/` -- Deformed mesh output
- `input_gcode/` -- Planar-sliced G-code
- `output_gcode/` -- Back-transformed 4-axis G-code
- `prusa_slicer/` -- PrusaSlicer configuration INI files
- `requirements.txt` -- Pinned Python dependencies

## Slicer CLI

All slicers share a standard CLI: `python main.py --stl <file> --model <name>`. Slicers that use PrusaSlicer also require `--prusa <path>`.

```bash
# Radial slicer
python main.py --stl model.stl --model mypart --prusa /path/to/PrusaSlicer

# Hybrid slicer (with optional tuning)
python main.py --stl model.stl --model mypart --prusa /path/to/PrusaSlicer \
  --max-overhang 25 --rotation-multiplier 1.5

# Conic slicer (cone-angle required)
python main.py --stl model.stl --model mypart --prusa /path/to/PrusaSlicer \
  --cone-angle 30 --z-split 1.0 --wave-type sine

# Generic slicer (--prusa optional, prompts for manual slicing if omitted)
python main.py --stl model.stl --model mypart --prusa /path/to/PrusaSlicer

# Native slicer (no PrusaSlicer needed, does its own implicit slicing)
python main.py --stl model.stl --model mypart --layer-height 0.2
```

Run `python main.py --help` in any slicer directory for full option details.

## Build Commands

### GUI (C++ / Qt6)

```bash
cd gui
cmake -B build
cmake --build build
```

Requires Qt6 (Widgets, 3DCore, 3DRender, 3DInput, 3DExtras, Quick). Post-build copies `slicerbundle/` to the executable directory.

### Python Slicers

Each Python slicer has its own virtual environment with pinned dependencies:

```bash
cd <slicer_dir>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --help
```

## Running Tests

### Smoke Tests (all slicers)

```bash
# From repo root, using any slicer venv with pytest installed
python -m pytest tests/ -m smoke -v
```

### Native Slicer Unit Tests

```bash
cd native_slicer
source venv/bin/activate
pytest tests/ -v            # all 22 tests
pytest tests/test_foo.py    # single test file
```

### CI

GitHub Actions runs both smoke tests and native slicer tests on push to `main`/`dev`/`test` and on PRs. See `.github/workflows/test.yml`.

## Key Python Dependencies

numpy, scipy, pyvista (mesh/visualization), networkx (graph algorithms), pygcode (G-code parsing), tetgen/open3d (3D geometry), robust_laplacian (vector fields), trimesh (mesh processing)

## Git Workflow

- `main` -- Stable, tested releases
- `test` -- Integration testing before merging to main
- `dev` -- Active development
- `slicer/<name>` -- Slicer-specific feature work
- `feature/<desc>` -- New features
- `fix/<desc>` -- Bug fixes
- `gui/<desc>` -- GUI work

Old branches are archived as `archive/<name>` tags.
