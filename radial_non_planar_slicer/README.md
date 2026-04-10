# Radial Non-Planar Slicer

Radial slicing with smooth blend between Cartesian (low Z) and radial (high Z) regions. Based on [jyjblrd/Radial_Non_Planar_Slicer](https://github.com/jyjblrd/Radial_Non_Planar_Slicer).

**Status:** Working

## Usage

```bash
source venv/bin/activate
python main.py --stl path/to/model.stl --model mymodel --prusa /path/to/PrusaSlicer
```

## CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--stl` | Yes | Path to input STL file |
| `--model` | Yes | Model name for output files |
| `--prusa` | Yes | Path to PrusaSlicer executable |

Deformation parameters (`angle_base`, `angle_factor`, `transition_z`, `blend_height`) are currently configured within `deform.py`.

## Pipeline

1. Load STL and apply radial deformation with Cartesian-to-radial blending
2. Planar slice deformed mesh with PrusaSlicer
3. Back-transform G-code to 4-axis coordinates

## Modules

- `main.py` - CLI entry point and pipeline orchestration
- `deform.py` - Radial deformation with transition blending
- `reform.py` - G-code back-transformation
- `better_visualizer.py` - Output visualization
- `visualize_gcode.py` - G-code path visualization

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
