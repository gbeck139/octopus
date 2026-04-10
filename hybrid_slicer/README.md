# Hybrid Slicer

Combines Cartesian and radial approaches using per-vertex rotation fields computed from surface overhang analysis with Laplacian smoothing.

**Status:** WIP

## Usage

```bash
source venv/bin/activate
python main.py --stl path/to/model.stl --model mymodel --prusa /path/to/PrusaSlicer
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--stl` | Yes | - | Path to input STL file |
| `--model` | Yes | - | Model name for output files |
| `--prusa` | Yes | - | Path to PrusaSlicer executable |
| `--max-overhang` | No | 25.0 | Overhang angle threshold in degrees |
| `--rotation-multiplier` | No | 1.5 | Scale factor for computed rotations |
| `--smoothing-iterations` | No | 30 | Laplacian smoothing passes |
| `--grid-resolution` | No | 1.0 | XY grid spacing (mm) for rotation lookup |
| `--max-pos-rotation` | No | 45.0 | Maximum positive rotation in degrees |
| `--max-neg-rotation` | No | -45.0 | Maximum negative rotation in degrees |

## Pipeline

1. Load STL and subdivide to target density
2. Analyze surface geometry to compute per-vertex rotation field
3. Deform mesh using the rotation field with Laplacian-smoothed transitions
4. Planar slice with PrusaSlicer
5. Back-transform G-code using stored rotation field

## Modules

- `main.py` - CLI entry point and pipeline orchestration
- `analyze.py` - Surface overhang analysis and rotation field computation
- `deform.py` - Per-column rotation-based mesh deformation
- `reform.py` - G-code back-transformation using rotation field

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
