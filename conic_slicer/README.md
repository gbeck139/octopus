# Conic Slicer

Conic deformation approach for 4-axis non-planar slicing. Splits the mesh at a configurable Z height, applies a conic transformation (`z' = z + cone_factor * r`) to the upper portion with optional wave modulation, then merges both halves after slicing.

**Status:** Active development

## Usage

```bash
source venv/bin/activate
python main.py --stl path/to/model.stl --model mymodel --prusa /path/to/PrusaSlicer --cone-angle 30 --z-split 5.0
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--stl` | Yes | - | Path to input STL file |
| `--model` | Yes | - | Model name for output files |
| `--prusa` | Yes | - | Path to PrusaSlicer executable |
| `--z-split` | No | 1.0 | Z height (mm) at which to split the mesh |
| `--cone-angle` | Yes | - | Cone half-angle in degrees |
| `--wave-type` | No | none | Wave modulation: none, sine, sawtooth, sine_curvature, sine_azimuthal, sine_normal |
| `--wave-amplitude` | No | 1.0 | Wave amplitude in mm |
| `--wave-length` | No | 5.0 | Wave wavelength in mm (or lobe count for sine_azimuthal) |

## Pipeline

1. Load STL, split at `--z-split` into lower and upper portions
2. Apply conic deformation to upper portion (with optional wave modulation)
3. Planar slice both portions with PrusaSlicer
4. Convert lower G-code to CXZB format, back-transform upper G-code
5. Merge both G-codes into final 4-axis output

## Modules

- `main.py` - CLI entry point and pipeline orchestration
- `deform.py` - Mesh loading, splitting, conic deformation, wave modulation
- `reform.py` - G-code back-transformation from planar to 4-axis coordinates
- `merge.py` - Merging lower (planar) and upper (conic) G-code files

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
