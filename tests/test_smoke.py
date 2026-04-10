"""
Smoke tests for all slicers.

Verifies that each slicer can be invoked with --help without errors.
Does NOT require PrusaSlicer, STL files, or any external dependencies
beyond what's in the slicer's venv.
"""

import subprocess
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_python(slicer_dir):
    """Find the Python executable in a slicer's venv."""
    for venv_name in ("venv", ".venv"):
        python = os.path.join(ROOT, slicer_dir, venv_name, "bin", "python3")
        if os.path.exists(python):
            return python
    pytest.skip(f"No venv found for {slicer_dir}")


def _run_help(slicer_dir, script="main.py"):
    """Run a slicer's main.py --help and assert it succeeds."""
    python = _find_python(slicer_dir)
    script_path = os.path.join(ROOT, slicer_dir, script)
    result = subprocess.run(
        [python, script_path, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{slicer_dir}/{script} --help failed:\n{result.stderr}"
    )
    return result.stdout


@pytest.mark.smoke
def test_conic_slicer_help():
    output = _run_help("conic_slicer")
    assert "--stl" in output
    assert "--cone-angle" in output


@pytest.mark.smoke
def test_radial_slicer_help():
    output = _run_help("radial_non_planar_slicer")
    assert "--stl" in output
    assert "--model" in output


@pytest.mark.smoke
def test_hybrid_slicer_help():
    output = _run_help("hybrid_slicer")
    assert "--stl" in output
    assert "--max-overhang" in output


@pytest.mark.smoke
def test_generic_slicer_help():
    output = _run_help("generic_non_planar_slicer")
    assert "--stl" in output
    assert "--prusa" in output
    assert "--visualize" in output


@pytest.mark.smoke
def test_native_slicer_help():
    output = _run_help("native_slicer")
    assert "--stl" in output
    assert "--model" in output


@pytest.mark.smoke
def test_native_slicer_imports():
    """Verify the cxzb_slicer package can be imported."""
    python = _find_python("native_slicer")
    result = subprocess.run(
        [python, "-c", "from cxzb_slicer.core.types import SlicerConfig; print('OK')"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=os.path.join(ROOT, "native_slicer"),
    )
    assert result.returncode == 0, (
        f"cxzb_slicer import failed:\n{result.stderr}"
    )
    assert "OK" in result.stdout
