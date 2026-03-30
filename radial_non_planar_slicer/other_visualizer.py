import sys
import os
import re
import argparse
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore

# Fix OpenGL issues in PyInstaller builds
os.environ["QT_OPENGL"] = "software"

class GCodeVisualizer:
    def __init__(self, filename, nozzle_offset=43):
        self.filename = filename
        self.nozzle_offset = nozzle_offset

        self.points = []
        self.is_extrusion = []
        self.poly = None
        self.plotter = None
        self.mesh_actor = None

        self.current_move_index = 0
        self.show_travel = False
        self.density_mode = False

        # required for commands
        self._cmd_buffer = ""

        self.load_gcode()

    def load_gcode(self):
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}", file=sys.stderr)
            return

        print(f"Reading {self.filename}...", file=sys.stderr)

        current_pos = {'X':0, 'Z':20, 'C':0, 'B':-15.0, 'E':0}
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
                    if tag == 'E':
                        if val > 0:
                            is_extru_move = True
                    else:
                        current_pos[tag] = val
                        changed = True

                if changed:
                    end_cartesian = self.machine_to_cartesian(current_pos)
                    points_list.append(end_cartesian)
                    is_ext.append(is_extru_move)

        self.points = np.array(points_list)
        self.is_extrusion = np.array(is_ext)
        self.current_move_index = len(self.is_extrusion)

        print(f"Loaded {len(self.points)} points, {len(self.is_extrusion)} lines.", file=sys.stderr)

    def machine_to_cartesian(self, params):
        x_gcode = params['X']
        z_gcode = params['Z']
        c_gcode = params['C']
        b_gcode = params['B']

        rotation_rad = np.deg2rad(-b_gcode)
        r_model = x_gcode - np.sin(rotation_rad) * self.nozzle_offset
        z_model = z_gcode - (np.cos(rotation_rad)-1)*self.nozzle_offset

        c_rad = np.deg2rad(c_gcode)
        x = r_model*np.cos(c_rad)
        y = r_model*np.sin(c_rad)
        z = z_model

        return [x, y, z]

    def create_scene(self):
        if len(self.points) < 2:
            return

        lines = np.empty((len(self.is_extrusion), 3), dtype=int)
        lines[:,0] = 2
        lines[:,1] = np.arange(len(self.is_extrusion))
        lines[:,2] = np.arange(1, len(self.is_extrusion)+1)

        self.poly = pv.PolyData()
        self.poly.points = self.points
        self.poly.lines = lines.flatten()
        self.poly.cell_data['MoveType'] = self.is_extrusion.astype(float)
        self.poly.cell_data['Index'] = np.arange(len(self.is_extrusion))

    def update_mesh(self):
        if self.poly is None:
            return

        subset = self.poly.threshold(
            value=[0, self.current_move_index],
            scalars='Index',
            preference='cell'
        )

        if not self.show_travel:
            subset = subset.threshold(
                value=[0.9, 1.1],
                scalars='MoveType',
                preference='cell'
            )

        if subset.n_cells == 0:
            if self.mesh_actor:
                self.plotter.remove_actor(self.mesh_actor)
                self.mesh_actor = None
            return

        opacity = 0.1 if self.density_mode else 1.0
        line_width = 1 if self.density_mode else 3
        render_tubes = False if self.density_mode else True

        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)

        self.mesh_actor = self.plotter.add_mesh(
            subset,
            scalars='MoveType',
            cmap=['red','lime'],
            clim=[0,1],
            line_width=line_width,
            render_lines_as_tubes=render_tubes,
            opacity=opacity,
            show_scalar_bar=False,
            reset_camera=False
        )

        self.plotter.render()

    def handle_command(self, line):
        line = line.strip()
        print("COMMAND RECEIVED:", line, file=sys.stderr)

        if line.startswith("MOVE"):
            self.current_move_index = int(line.split()[1])
            self.update_mesh()
        elif line.startswith("TOGGLE_TRAVEL"):
            self.show_travel = bool(int(line.split()[1]))
            self.update_mesh()
        elif line.startswith("TOGGLE_DENSITY"):
            self.density_mode = bool(int(line.split()[1]))
            self.update_mesh()

    # ✅ Simplified and working for buttons
    def poll_stdin(self):
        import msvcrt
        import sys

        while msvcrt.kbhit():
            line = sys.stdin.readline()
            if line:
                self.handle_command(line)

    def visualize(self):
        if self.poly is None:
            self.create_scene()

        app = QtWidgets.QApplication(sys.argv)
        window = QtWidgets.QMainWindow()

        self.plotter = QtInteractor(window)
        window.setCentralWidget(self.plotter)

        window.resize(800, 600)
        window.show()

        self.update_mesh()

        # ONLY stdout output for embedding
        print(int(window.winId()), flush=True)

        sys.stdin = open(0, 'r', buffering=1)

        timer = QtCore.QTimer()
        timer.timeout.connect(self.poll_stdin)
        timer.start(50)

        app.exec_()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcode", required=True)
    args = parser.parse_args()

    viz = GCodeVisualizer(args.gcode)
    viz.visualize()


if __name__ == "__main__":
    main()