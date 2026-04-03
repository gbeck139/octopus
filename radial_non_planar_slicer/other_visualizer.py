import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
import re
import os
import argparse
import sys


class GCodeVisualizer:
    def __init__(self, filename, nozzle_offset=43):
        self.filename = filename
        self.nozzle_offset = nozzle_offset
        self.points = []
        self.is_extrusion = []
        self.poly = None
        self.plotter = None
        self.mesh_actor = None

        # State
        self.current_move_index = 0
        self.show_travel = False
        self.density_mode = False

        self.load_gcode()

    def load_gcode(self):
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}")
            return

        current_pos = {'X': 0, 'Z': 20, 'C': 0, 'B': -15.0, 'E': 0}
        points_list = []
        is_ext = []

        start_cartesian = self.machine_to_cartesian(current_pos)
        points_list.append(start_cartesian)

        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                if not (line.startswith('G0') or line.startswith('G1')):
                    continue

                matches = re.findall(r'([XZCBE])([-\d\.]+)', line)
                if not matches:
                    continue

                is_extru_move = False
                changed = False

                for tag, val_str in matches:
                    val = float(val_str)
                    if tag == 'E' and val > 0:
                        is_extru_move = True
                    else:
                        current_pos[tag] = val
                        changed = True

                if changed:
                    points_list.append(self.machine_to_cartesian(current_pos))
                    is_ext.append(is_extru_move)

        self.points = np.array(points_list)
        self.is_extrusion = np.array(is_ext)
        self.current_move_index = len(self.is_extrusion)

    def machine_to_cartesian(self, params):
        x = params['X']
        z = params['Z']
        c = params['C']
        b = params['B']

        rotation_rad = np.deg2rad(-b)
        r_model = x - np.sin(rotation_rad) * self.nozzle_offset
        z_model = z - (np.cos(rotation_rad) - 1) * self.nozzle_offset
        c_rad = np.deg2rad(c)

        return [r_model * np.cos(c_rad), r_model * np.sin(c_rad), z_model]

    def create_scene(self):
        if len(self.points) < 2:
            return

        num_lines = len(self.is_extrusion)
        lines = np.empty((num_lines, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(num_lines)
        lines[:, 2] = np.arange(1, num_lines + 1)

        self.poly = pv.PolyData()
        self.poly.points = self.points
        self.poly.lines = lines.flatten()
        self.poly.cell_data['MoveType'] = self.is_extrusion.astype(float)
        self.poly.cell_data['Index'] = np.arange(num_lines)

    def update_mesh(self):
        if self.poly is None or self.plotter is None:
            return

        subset = self.poly.threshold(value=[0, self.current_move_index],
                                     scalars='Index', preference='cell')

        if not self.show_travel:
            subset = subset.threshold(value=[0.9, 1.1],
                                      scalars='MoveType', preference='cell')

        if subset.n_cells == 0:
            if self.mesh_actor:
                self.plotter.remove_actor(self.mesh_actor)
                self.mesh_actor = None
            return

        opacity = 0.1 if self.density_mode else 1.0
        line_width = 1 if self.density_mode else 3
        render_tubes = not self.density_mode

        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)

        self.mesh_actor = self.plotter.add_mesh(
            subset, scalars='MoveType', cmap=['red', 'lime'], clim=[0, 1],
            line_width=line_width, render_lines_as_tubes=render_tubes,
            opacity=opacity, show_scalar_bar=False, reset_camera=False
        )

    # --- COMMANDS FROM C++ ---
    def slider_callback(self, value):
        self.current_move_index = int(value)
        self.update_mesh()

    def toggle_travel(self, flag):
        self.show_travel = flag
        self.update_mesh()

    def toggle_density(self, flag):
        self.density_mode = flag
        self.update_mesh()

    # --- PROCESS COMMAND LOOP ---
    def process_commands(self):
        try:
            line = sys.stdin.readline()
            if not line:
                return
            parts = line.strip().split()
            if not parts:
                return
            cmd = parts[0].upper()
            if cmd == "SLIDER":
                self.slider_callback(int(parts[1]))
            elif cmd == "TRAVEL":
                self.toggle_travel(bool(int(parts[1])))
            elif cmd == "DENSITY":
                self.toggle_density(bool(int(parts[1])))
        except Exception:
            pass  # pipe timing issues

    # --- VISUALIZATION ---
    def visualize(self):
        self.create_scene()
        if self.poly is None:
            return

        app = QtWidgets.QApplication(sys.argv)

        window = QtWidgets.QMainWindow()
        self.plotter = QtInteractor(window)
        window.setCentralWidget(self.plotter)

        # Make window frameless for embedding
        window.setWindowFlag(Qt.Window, False)
        window.setWindowFlag(Qt.FramelessWindowHint, True)
        window.show()
        window.winId()  # Force native creation
        print(int(window.winId()))
        sys.stdout.flush()
        window.setGeometry(-10000, -10000, 800, 600)
        window.show()
        #window.hide()  # Hide until embedded

        self.plotter.add_text(os.path.basename(self.filename))
        self.update_mesh()

        # Timer to poll commands
        timer = QTimer()
        timer.timeout.connect(self.process_commands)
        timer.start(30)

        # Run **Qt event loop without blocking** for embedding
        app.processEvents()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcode", required=True)
    args = parser.parse_args()

    viz = GCodeVisualizer(args.gcode)
    viz.visualize()


if __name__ == "__main__":
    main()