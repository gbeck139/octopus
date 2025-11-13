import numpy as np
import pyvista as pv
import networkx as nx
from pygcode import Line
import time
from scipy.spatial.transform import Rotation as R
import deform
import reform


MODEL_NAME = 'propeller'  # name of the model file without extension

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, ROTATION, offsets_applied = deform.deform_mesh(mesh)
deform.save_deformed_mesh(deformed_mesh, MODEL_NAME)
# deform.plot_deformed_mesh(deformed_mesh)
# input("Press Enter after slicing is complete...")
reform.load_gcode_and_undeform(MODEL_NAME, ROTATION, offsets_applied)