"""Ensure cxzb_slicer is importable regardless of how pytest resolves rootdir."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
