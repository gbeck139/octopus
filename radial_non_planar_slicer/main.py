import numpy as np
import pyvista as pv
import networkx as nx
from pygcode import Line
import time
from scipy.spatial.transform import Rotation as R
import deform
import reform


#TODO separate reform and deform and make command with arguments

MODEL_NAME = '3DBenchy'

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, transform_params = deform.deform_mesh(mesh)
# deform.save_deformed_mesh(deformed_mesh, transform_params, MODEL_NAME)
# deform.plot_deformed_mesh(deformed_mesh)
# input("Press Enter after slicing is complete...")
#TODO call subprocess to run slicer here
"""
subprocess.run([
    "orca-slicer",
    "--load", "profile.ini",
    "--input", config["paths"]["deformed_stl"],
    "--output", config["paths"]["deformed_gcode"]
], check=True)
"""
reform.load_gcode_and_undeform(MODEL_NAME, transform_params)
