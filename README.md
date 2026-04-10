# Octopus

Multi-strategy 4-axis (XYZC) printer slicing system for non-planar additive manufacturing. Targets CX-ZB kinematics printers (XYZ + rotational C axis) and provides multiple slicing strategies for handling overhangs through non-planar layer approaches.

Originally based on [jyjblrd/Radial_Non_Planar_Slicer](https://github.com/jyjblrd/Radial_Non_Planar_Slicer).

## Core Pipeline

All slicers follow a three-stage pipeline: **Deform -> Slice -> Reform**

1. **Deform** - Transform 3D mesh into modified geometry suitable for planar slicing
2. **Slice** - Use PrusaSlicer on the deformed mesh (standard planar slicing)
3. **Reform** - Back-transform the resulting G-code into 4-axis (XYZC) coordinates

## Slicing Strategies

| Slicer | Strategy | Status |
|--------|----------|--------|
| [conic_slicer](conic_slicer/) | Conic deformation (`z' = z + cone_factor * r`) with mesh splitting and wave modulation | Active development |
| [radial_non_planar_slicer](radial_non_planar_slicer/) | Smooth blend between Cartesian (low Z) and radial (high Z) regions | Working |
| [hybrid_slicer](hybrid_slicer/) | Per-vertex rotation fields from surface overhang analysis with Laplacian smoothing | WIP |
| [generic_non_planar_slicer](generic_non_planar_slicer/) | Tetrahedral mesh deformation using tetgen, NetworkX, and scipy optimization | Experimental |
| [native_slicer](native_slicer/) | Modular implicit slicing pipeline with full IK, vector fields, and toolpath generation | WIP (has tests) |

## Prerequisites

- **Python 3.12+**
- **PrusaSlicer** (CLI) - required by conic, radial, hybrid, and generic slicers
- **Qt6** - required for building the GUI (Widgets, 3DCore, 3DRender, 3DInput, 3DExtras, Quick)
- **CMake** - required for building the GUI

## Quick Start

Each slicer has its own virtual environment and requirements:

```bash
cd <slicer_dir>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --help
```

See each slicer's README for specific usage and CLI arguments.

## GUI

```bash
cd gui
cmake -B build
cmake --build build
```

The GUI provides a tab-based interface (Prepare, Preview, Monitor) with a setup wizard for configuring paths to PrusaSlicer, Python, and printer config.

## Repository Structure

```
octopus/
├── conic_slicer/                  # Conic deformation slicer
├── radial_non_planar_slicer/      # Radial blend slicer
├── hybrid_slicer/                 # Overhang-driven rotation slicer
├── generic_non_planar_slicer/     # Tetrahedral deformation slicer
├── native_slicer/                 # Modular cxzb_slicer package
│   ├── cxzb_slicer/               # Python package (core, kinematics, layers, etc.)
│   ├── tests/                     # pytest test suite (22 tests)
│   └── examples/                  # Usage examples
├── gui/                           # Qt6 C++ desktop application
├── firmware/                      # Printer firmware and Duet config
├── tests/                         # Top-level smoke tests
└── .github/workflows/             # CI/CD
```

Each slicer directory follows a standard I/O layout:
- `input_models/` - STL input files
- `output_models/` - Deformed mesh output + transform metadata (JSON)
- `input_gcode/` - Planar-sliced G-code from PrusaSlicer
- `output_gcode/` - Back-transformed 4-axis G-code

## Printer Specifications

| Axis | Range |
|------|-------|
| X | -163 to 163 mm |
| Z | 0 to 280 mm |
| C | Infinite (continuous rotation) |
| B | -170 to 170 degrees |

## Branch Convention

- `main` - Stable, tested releases
- `dev` - Integration branch
- `slicer/<name>` - Slicer-specific feature work
- `feature/<desc>` - New features
- `fix/<desc>` - Bug fixes
- `gui/<desc>` - GUI work

## Running Tests

```bash
# Smoke tests (verify imports and CLI across all slicers)
source conic_slicer/venv/bin/activate
pip install pytest
pytest tests/ -m smoke -v

# Native slicer unit tests
cd native_slicer
source venv/bin/activate
pytest tests/ -v
```
