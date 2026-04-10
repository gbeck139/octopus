# Native Slicer (cxzb_slicer)

Modular Python package for implicit slicing, vector-field-based orientation, CX-ZB inverse kinematics with RTCP compensation, extrusion planning, and G-code generation. This is the most architecturally mature slicer — designed as a library rather than a script.

**Status:** WIP (has test suite)

## Package Structure

```
cxzb_slicer/
├── core/           # Types (Layer, SlicerConfig, TOOLPATH_DTYPE) and configuration
├── kinematics/     # CX-ZB inverse kinematics solver, singularity handling
├── layers/         # Layer generators: planar, conic, spherical, contour extraction
├── vector_field/   # Poisson-based field solver, constraints, smoothing
├── sdf/            # Signed distance field grid, mesh-to-SDF providers
├── toolpath/       # Contour-to-path, rectilinear infill, nearest-neighbor optimizer
├── gcode/          # G-code writer (Duet/Marlin), validator with FK roundtrip
├── extrusion/      # Tilt compensation, retraction planning
└── visualization/  # PyVista-based viewer
```

## Usage

```python
from cxzb_slicer.kinematics import CXZBSolver
from cxzb_slicer.layers import PlanarLayerGenerator
from cxzb_slicer.core.types import SlicerConfig
```

See `examples/` for complete slicing workflows (slice_cube.py, slice_conic.py, slice_sphere.py).

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

22 tests covering kinematics roundtrips, G-code formatting, SDF sampling, layer generation, toolpath optimization, vector field computation, and extrusion compensation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
