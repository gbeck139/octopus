# Generic Non-Planar Slicer

Tetrahedral mesh approach using tetgen for tetrahedralization, NetworkX for graph algorithms, and scipy optimization for mesh deformation. This is the most general approach — it can handle arbitrary deformation fields.

**Status:** Experimental

## Usage

```bash
source venv/bin/activate
python main.py --model propeller --stl path/to/model.stl --visualize
```

## CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | No | propeller | Model name |
| `--stl` | No | - | Path to input STL (uses input_models/ if not specified) |
| `--slicer` | No | - | Path to slicer executable (not yet wired) |
| `--slicer-config` | No | - | Path to slicer config (not yet wired) |
| `--visualize` | No | false | Show visualization during processing |

## Dependencies

This slicer has the heaviest dependency set (86 packages) including tetgen, open3d, dash, and scipy.

## Known Limitations

- Single monolithic `main.py` (68KB) — functional but hard to test in isolation
- Module-level argparse execution makes importing difficult
- `--slicer` and `--slicer-config` args are defined but not yet connected

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
