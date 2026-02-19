import numpy as np
import pyvista as pv
import re
import os

class GCodeVisualizer:
    def __init__(self, filename, nozzle_offset=43):
        self.filename = filename
        self.nozzle_offset = nozzle_offset
        self.points = []
        self.is_extrusion = []
        self.poly = None
        self.plotter = None
        self.mesh_actor = None
        
        # Initialize state variables
        self.current_move_index = 0
        self.show_travel = False
        self.density_mode = False

        self.load_gcode()

    def load_gcode(self):
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}")
            return

        print(f"Reading {self.filename}...")
        
        raw_points = []
        is_ext = []
        
        # Initial position (Home)
        # G0 C0 X0 Z20 B-15.0
        current_pos = {'X': 0, 'Z': 20, 'C': 0, 'B': -15.0, 'E': 0}
        
        # Parse file
        # We need to track moves.
        # A move is defined by a G0/G1 line.
        # From Start Point -> End Point.
        
        # Process:
        # 1. Start at Home.
        # 2. Convert Home to Cartesian -> P0
        # 3. Read Line. Update Machine Coords -> Current Machine Pos.
        # 4. Convert Current Machine Pos -> Cartesian -> P1.
        # 5. Segment is P0->P1.
        # 6. P0 = P1.
        
        points_list = []
        
        # Add initial point
        start_cartesian = self.machine_to_cartesian(current_pos)
        points_list.append(start_cartesian)
        
        with open(self.filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                if not (line.startswith('G0') or line.startswith('G1')):
                    continue
                
                # Parse Params
                # Helper dictionary for this line
                line_params = {}
                
                # Simple regex sufficient for standard GCode
                # Find all letter-float pairs
                matches = re.findall(r'([XZCBE])([-\d\.]+)', line)
                if not matches:
                    continue
                    
                is_extru_move = False
                
                # Update current_pos
                changed = False
                e_val = 0
                
                for tag, val_str in matches:
                    val = float(val_str)
                    if tag == 'E':
                        e_val = val
                        # In relative mode (M83), E>0 is extrusion.
                        # We assume relative extrusion as per header comments
                        if val > 0:
                            is_extru_move = True
                    else:
                        current_pos[tag] = val
                        changed = True
                
                # Only add point if position changed (ignore pure E moves if no XYZCB change? 
                # Actually pure E moves are retractions/primes at same location. 
                # We usually don't visualize them as lines unless they move.
                # If XYZCB didn't change, start==end, line length 0.
                
                if changed:
                    end_cartesian = self.machine_to_cartesian(current_pos)
                    points_list.append(end_cartesian)
                    is_ext.append(is_extru_move)

        self.points = np.array(points_list)
        self.is_extrusion = np.array(is_ext)
        print(f"Loaded {len(self.points)} points, {len(self.is_extrusion)} lines.")
        self.current_move_index = len(self.is_extrusion)

    def machine_to_cartesian(self, params):
        x_gcode = params['X']
        z_gcode = params['Z']
        c_gcode = params['C']
        b_gcode = params['B']
        
        # 1. Recover Rotation (B is negative degrees of rotation)
        rotation_rad = np.deg2rad(-b_gcode)
        
        # 2. Reverse Nozzle Offset
        r_model = x_gcode - np.sin(rotation_rad) * self.nozzle_offset
        z_model = z_gcode - (np.cos(rotation_rad) - 1) * self.nozzle_offset
        
        # 3. Polar to Cartesian
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

        # Connectivity array for individual segments
        # PyVista standard: [n_pts, p1, p2, n_pts, p3, p4, ...]
        # We use 2 points per line segment
        
        lines = np.empty((num_lines, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(num_lines)      # Start point index
        lines[:, 2] = np.arange(1, num_lines+1) # End point index
        
        self.poly = pv.PolyData()
        self.poly.points = self.points
        self.poly.lines = lines.flatten()
        
        # Add scalars for coloring
        # 0 for travel, 1 for extrusion
        self.poly.cell_data['MoveType'] = self.is_extrusion.astype(float)
        
        # Add Index for filtering history
        self.poly.cell_data['Index'] = np.arange(num_lines)

    def update_mesh(self, obj=None, event=None):
        # Callback wrapper specific to PyVista widgets? 
        # Actually standard python method called by our own wrappers
        
        if self.poly is None: return
        
        # Extract subset up to current index
        # We want cells with 'Index' <= current_move_index
        # threshold is inclusive.
        
        # Strategy:
        # 1. Threshold by Index (History)
        # 2. Threshold by MoveType (Travel Filter)
        
        subset = self.poly.threshold(value=[0, self.current_move_index], scalars='Index', preference='cell')
        
        if not self.show_travel:
            # Travel is 0. Extrusion is 1.
            # Range [0.9, 1.1] keeps only 1.
            subset = subset.threshold(value=[0.9, 1.1], scalars='MoveType', preference='cell')
            
        if subset.n_cells == 0:
            if self.mesh_actor: 
                self.plotter.remove_actor(self.mesh_actor)
                self.mesh_actor = None
            return

        opacity = 0.1 if self.density_mode else 1.0
        line_width = 1 if self.density_mode else 3
        render_tubes = False if self.density_mode else True
        
        # Coloring
        # If density mode, maybe just one color (Cyan?) to see density better?
        # Or keep extrusion color (Lime).
        
        # cmap: Red=0 (Travel), Lime=1 (Extrude)
        
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)
            
        self.mesh_actor = self.plotter.add_mesh(
            subset,
            scalars='MoveType',
            cmap=['red', 'lime'],
            clim=[0, 1],
            line_width=line_width,
            render_lines_as_tubes=render_tubes,
            opacity=opacity,
            show_scalar_bar=False,
            reset_camera=False
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

    def visualize(self):
        if self.poly is None:
            self.create_scene()
            if self.poly is None: return
            
        self.plotter = pv.Plotter()
        self.plotter.add_text(f"Visualizing: {os.path.basename(self.filename)}", position='upper_left')
        
        # Initial Render
        self.update_mesh()
        
        # Add Slider
        self.plotter.add_slider_widget(
            self.slider_callback,
            [0, len(self.is_extrusion)-1],
            title='History',
            value=len(self.is_extrusion)-1,
            pointa=(0.4, 0.9),
            pointb=(0.9, 0.9),
            fmt="%0.f"
        )
        
        # Add Checkboxes
        # Travel (Bool)
        # Note: PyVista checkbox widgets require keeping a reference if creating manually, 
        # but add_checkbox_button_widget manages it.
        
        # 1. Show Travel
        self.plotter.add_checkbox_button_widget(
            self.toggle_travel,
            value=False,
            position=(10, 10),
            size=30,
            border_size=2,
            color_on='green',
            color_off='grey'
        )
        self.plotter.add_text("Travel", position=(50, 15), font_size=12)

        # 2. Density Mode
        self.plotter.add_checkbox_button_widget(
            self.toggle_density,
            value=False,
            position=(10, 50),
            size=30,
            border_size=2,
            color_on='blue',
            color_off='grey'
        )
        self.plotter.add_text("Density", position=(50, 55), font_size=12)

        self.plotter.show()

if __name__ == "__main__":
    FILENAME = "radial_non_planar_slicer/output_gcode/3DBenchy_reformed.gcode"
    viz = GCodeVisualizer(FILENAME)
    viz.visualize()
