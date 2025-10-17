import numpy as np
import pyvista as pv
import networkx as nx
from pygcode import Line
import time
from scipy.spatial.transform import Rotation as R
import deform


MODEL_NAME = 'motos_para_a_eternidade'

mesh = deform.load_mesh(MODEL_NAME)
deformed_mesh, ROTATION, offsets_applied = deform.deform_mesh(mesh)
deform.save_deformed_mesh(deformed_mesh, MODEL_NAME)
deform.plot_deformed_mesh(deformed_mesh)
