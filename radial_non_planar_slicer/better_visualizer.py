import argparse
import os
import re

import numpy as np
import pyvista as pv


class GCodeVisualizer:
    def __init__(self, filename, nozzle_offset=43, color_mode="movetype"):
        self.filename = filename
        self.nozzle_offset = nozzle_offset
        self.points = []
        self.is_extrusion = []
        self.b_values = []
        self.bc_per_point = []
        self.poly = None
        self.plotter = None
        self.mesh_actor = None
        self.travel_actor = None
        self.head_actor = None

        self.current_move_index = 0
        self.show_travel = False
        self.density_mode = False

        self.color_modes = ["MoveType", "ZHeight", "Index", "Tilt"]
        self.color_mode = color_mode.capitalize() if color_mode != "movetype" else "MoveType"
        if self.color_mode == "Zheight":
            self.color_mode = "ZHeight"
        if self.color_mode not in self.color_modes:
            self.color_mode = "MoveType"
        self.color_text_actor = None

        self.animation_playing = False
        self.animation_speed = 100
        self.animation_timer_id = None
        self.status_text_actor = None
        self.help_text_actor = None
        self.slider_widget = None

        self.load_gcode()

    def load_gcode(self):
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}")
            return

        print(f"Reading {self.filename}...")

        is_ext = []
        b_vals = []
        bc_points = []

        current_pos = {"X": 0, "Z": 20, "C": 0, "B": -15.0, "E": 0}
        points_list = []

        start_cartesian = self.machine_to_cartesian(current_pos)
        points_list.append(start_cartesian)
        bc_points.append((current_pos["B"], current_pos["C"]))

        with open(self.filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):
                    continue

                if not (line.startswith("G0") or line.startswith("G1")):
                    continue

                matches = re.findall(r"([XZCBE])([-\d\.]+)", line)
                if not matches:
                    continue

                is_extru_move = False
                changed = False

                for tag, val_str in matches:
                    val = float(val_str)
                    if tag == "E":
                        if val > 0:
                            is_extru_move = True
                    else:
                        current_pos[tag] = val
                        changed = True

                if changed:
                    end_cartesian = self.machine_to_cartesian(current_pos)
                    points_list.append(end_cartesian)
                    is_ext.append(is_extru_move)
                    b_vals.append(current_pos["B"])
                    bc_points.append((current_pos["B"], current_pos["C"]))

        self.points = np.array(points_list)
        self.is_extrusion = np.array(is_ext)
        self.b_values = np.array(b_vals)
        self.bc_per_point = bc_points
        print(f"Loaded {len(self.points)} points, {len(self.is_extrusion)} lines.")
        self.current_move_index = len(self.is_extrusion)

    def machine_to_cartesian(self, params):
        x_gcode = params["X"]
        z_gcode = params["Z"]
        c_gcode = params["C"]
        b_gcode = params["B"]

        rotation_rad = np.deg2rad(-b_gcode)
        r_model = x_gcode - np.sin(rotation_rad) * self.nozzle_offset
        z_model = z_gcode - (np.cos(rotation_rad) - 1) * self.nozzle_offset
        c_rad = np.deg2rad(c_gcode)

        x = r_model * np.cos(c_rad)
        y = r_model * np.sin(c_rad)
        z = z_model
        return [x, y, z]

    def create_scene(self):
        num_points = len(self.points)
        num_lines = len(self.is_extrusion)
        if num_points < 2:
            return

        lines = np.empty((num_lines, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(num_lines)
        lines[:, 2] = np.arange(1, num_lines + 1)

        self.poly = pv.PolyData()
        self.poly.points = self.points
        self.poly.lines = lines.flatten()
        self.poly.cell_data["MoveType"] = self.is_extrusion.astype(float)
        self.poly.cell_data["Index"] = np.arange(num_lines).astype(float)

        z_start = self.points[:-1, 2]
        z_end = self.points[1:, 2]
        self.poly.cell_data["ZHeight"] = ((z_start + z_end) / 2.0).astype(float)
        self.poly.cell_data["Tilt"] = self.b_values.astype(float)

    def _get_color_settings(self):
        if self.color_mode == "MoveType":
            return "MoveType", ["red", "lime"], [0, 1], False
        if self.color_mode == "ZHeight":
            data = self.poly.cell_data["ZHeight"]
            return "ZHeight", "viridis", [data.min(), data.max()], True
        if self.color_mode == "Index":
            data = self.poly.cell_data["Index"]
            return "Index", "rainbow", [0, data.max()], True
        if self.color_mode == "Tilt":
            data = self.poly.cell_data["Tilt"]
            return "Tilt", "coolwarm", [data.min(), data.max()], True
        return "MoveType", ["red", "lime"], [0, 1], False

    def update_mesh(self, obj=None, event=None):
        if self.poly is None:
            return

        self.plotter.suppress_rendering = True
        subset = self.poly.threshold(value=[0, self.current_move_index], scalars="Index", preference="cell")

        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)
            self.mesh_actor = None
        if self.travel_actor:
            self.plotter.remove_actor(self.travel_actor)
            self.travel_actor = None

        scalars, cmap, clim, show_sbar = self._get_color_settings()

        if self.color_mode != "MoveType" and self.show_travel:
            travel_sub = subset.threshold(value=[0, 0.1], scalars="MoveType", preference="cell")
            extrude_sub = subset.threshold(value=[0.9, 1.1], scalars="MoveType", preference="cell")

            if travel_sub.n_cells > 0:
                self.travel_actor = self.plotter.add_mesh(
                    travel_sub,
                    color="gray",
                    line_width=1,
                    render_lines_as_tubes=False,
                    opacity=0.3,
                    show_scalar_bar=False,
                    reset_camera=False,
                )
            subset = extrude_sub
        elif not self.show_travel:
            subset = subset.threshold(value=[0.9, 1.1], scalars="MoveType", preference="cell")

        if subset.n_cells == 0:
            if hasattr(self.plotter, "scalar_bars"):
                self.plotter.scalar_bars.clear()
            self._update_toolhead()
            self.plotter.suppress_rendering = False
            return

        opacity = 0.1 if self.density_mode else 1.0
        line_width = 1 if self.density_mode else 3
        render_tubes = not self.density_mode

        if hasattr(self.plotter, "scalar_bars"):
            self.plotter.scalar_bars.clear()

        self.mesh_actor = self.plotter.add_mesh(
            subset,
            scalars=scalars,
            cmap=cmap,
            clim=clim,
            line_width=line_width,
            render_lines_as_tubes=render_tubes,
            opacity=opacity,
            show_scalar_bar=show_sbar,
            scalar_bar_args={"title": self.color_mode} if show_sbar else None,
            reset_camera=False,
        )

        self._update_toolhead()
        self.plotter.suppress_rendering = False

    def _update_toolhead(self):
        if self.head_actor:
            self.plotter.remove_actor(self.head_actor)
            self.head_actor = None

        idx = min(self.current_move_index, len(self.points) - 1)
        tip = self.points[idx]
        b_deg, c_deg = self.bc_per_point[idx]
        rotation_rad = np.deg2rad(-b_deg)
        c_rad = np.deg2rad(c_deg)

        nozzle_len = 10.0
        dr = np.sin(rotation_rad) * nozzle_len
        dz = np.cos(rotation_rad) * nozzle_len
        dx = dr * np.cos(c_rad)
        dy = dr * np.sin(c_rad)
        base = tip + np.array([dx, dy, dz])

        nozzle_line = pv.Line(tip, base)
        self.head_actor = self.plotter.add_mesh(
            nozzle_line,
            color="cyan",
            line_width=5,
            render_lines_as_tubes=True,
            show_scalar_bar=False,
            reset_camera=False,
        )

    def slider_callback(self, value):
        self.current_move_index = int(value)
        self.update_mesh()

    def toggle_travel(self, flag):
        self.show_travel = flag
        self.update_mesh()

    def toggle_density(self, flag):
        self.density_mode = flag
        self.update_mesh()

    def _cycle_color_mode(self):
        idx = self.color_modes.index(self.color_mode)
        self.color_mode = self.color_modes[(idx + 1) % len(self.color_modes)]
        self._update_color_text()
        self.update_mesh()

    def _update_color_text(self):
        if self.color_text_actor:
            self.plotter.remove_actor(self.color_text_actor)
        self.color_text_actor = self.plotter.add_text(
            f"Color: {self.color_mode}",
            position="upper_right",
            font_size=10,
            color="white",
            name="color_mode_text",
        )

    def _update_status_text(self):
        if self.status_text_actor:
            self.plotter.remove_actor(self.status_text_actor)
        if self.animation_playing:
            text = f"PLAYING [{self.animation_speed}x]"
        else:
            text = "PAUSED"
        self.status_text_actor = self.plotter.add_text(
            text,
            position=(10, 90),
            font_size=10,
            color="yellow",
            name="status_text",
        )

    def _on_timer(self, obj, event):
        if not self.animation_playing:
            return

        max_idx = len(self.is_extrusion)
        self.current_move_index = min(self.current_move_index + self.animation_speed, max_idx)

        if self.slider_widget is not None:
            self.slider_widget.GetRepresentation().SetValue(self.current_move_index)

        self.update_mesh()

        if self.current_move_index >= max_idx:
            self.animation_playing = False
            self._update_status_text()

        self.plotter.render()

    def _toggle_play_pause(self):
        if self.animation_playing:
            self.animation_playing = False
        else:
            if self.current_move_index >= len(self.is_extrusion):
                self.current_move_index = 0
            self.animation_playing = True
            if self.animation_timer_id is None:
                iren = self.plotter.iren.interactor
                self.animation_timer_id = iren.CreateRepeatingTimer(33)
                iren.AddObserver("TimerEvent", self._on_timer)
        self._update_status_text()

    def _speed_up(self):
        self.animation_speed = min(self.animation_speed * 2, 100000)
        self._update_status_text()

    def _speed_down(self):
        self.animation_speed = max(self.animation_speed // 2, 1)
        self._update_status_text()

    def _reset_animation(self):
        self.animation_playing = False
        self.current_move_index = 0
        if self.slider_widget is not None:
            self.slider_widget.GetRepresentation().SetValue(0)
        self.update_mesh()
        self._update_status_text()

    def visualize(self):
        if self.poly is None:
            self.create_scene()
            if self.poly is None:
                return

        self.plotter = pv.Plotter()
        self.plotter.add_text(f"Visualizing: {os.path.basename(self.filename)}", position="upper_left")

        self.update_mesh()
        self._update_color_text()
        self._update_status_text()
        self.help_text_actor = self.plotter.add_text(
            "c: color mode | Space: play/pause | Up/Down: speed | r: reset",
            position=(10, 120),
            font_size=8,
            color="lightgray",
            name="help_text",
        )

        self.slider_widget = self.plotter.add_slider_widget(
            self.slider_callback,
            [0, len(self.is_extrusion) - 1],
            title="History",
            value=len(self.is_extrusion) - 1,
            pointa=(0.4, 0.9),
            pointb=(0.9, 0.9),
            fmt="%0.f",
        )

        self.plotter.add_checkbox_button_widget(
            self.toggle_travel,
            value=False,
            position=(10, 10),
            size=30,
            border_size=2,
            color_on="green",
            color_off="grey",
        )
        self.plotter.add_text("Travel", position=(50, 15), font_size=12)

        self.plotter.add_checkbox_button_widget(
            self.toggle_density,
            value=False,
            position=(10, 50),
            size=30,
            border_size=2,
            color_on="blue",
            color_off="grey",
        )
        self.plotter.add_text("Density", position=(50, 55), font_size=12)

        self.plotter.add_key_event("c", lambda: self._cycle_color_mode())
        self.plotter.add_key_event("space", lambda: self._toggle_play_pause())
        self.plotter.add_key_event(" ", lambda: self._toggle_play_pause())
        self.plotter.add_key_event("Up", lambda: self._speed_up())
        self.plotter.add_key_event("Down", lambda: self._speed_down())
        self.plotter.add_key_event("r", lambda: self._reset_animation())

        self.plotter.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcode", required=True)
    parser.add_argument(
        "--color-mode",
        default="movetype",
        choices=["movetype", "zheight", "index", "tilt"],
        help="Initial color mode",
    )
    args = parser.parse_args()

    viz = GCodeVisualizer(args.gcode, color_mode=args.color_mode)
    viz.visualize()


if __name__ == "__main__":
    main()
