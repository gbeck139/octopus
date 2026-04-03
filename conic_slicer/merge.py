import os
import sys

if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_GCODE_DIR = os.path.join(base_dir, "input_gcode")
OUTPUT_GCODE_DIR = os.path.join(base_dir, "output_gcode")


def write_header(fh):
    """Write the printer initialization G-code."""
    fh.write("; --- INITIALIZATION ---\n")
    fh.write("G21              ; Establish metric units (millimeters)\n")
    fh.write("G90              ; Use absolute coordinates for all axes\n")
    fh.write("G94              ; Set feedrate mode to units per minute (mm/min)\n")
    fh.write("M106 S128        ; Enable cooling fan at 50% power (PWM 128/255)\n")
    fh.write("; --- THERMAL MANAGEMENT ---\n")
    fh.write("M104 S200        ; Start heating nozzle to 200°C (Non-blocking)\n")
    fh.write("M109 S200        ; Wait for nozzle to reach target temperature before proceeding\n")
    fh.write("; --- HOMING & INITIAL POSITIONING ---\n")
    fh.write("G28              ; Execute homing sequence for all axes\n")
    fh.write("G0 C0 X0 Z20 B-15.0 F600 ; Rapid move to safe start clearance and orientation\n")
    fh.write("; --- EXTRUDER PREPARATION ---\n")
    fh.write("G92 E0           ; Zero out the current extruder position\n")
    fh.write("G1 E10 F200      ; Perform 10mm purge to prime the nozzle\n")
    fh.write("G92 E0           ; Reset extruder position after priming\n")
    fh.write("M83              ; Switch to relative extrusion mode\n")


def write_footer(fh):
    """Write the printer shutdown G-code."""
    fh.write("M104 S0          ; Disable nozzle heater (Allow to cool)\n")
    fh.write("M106 S0          ; Turn off component cooling fan\n")
    fh.write("M84              ; Cut power to stepper motors\n")


def merge(planar_cxzb_path, conic_gcode_path, model_name):
    """
    Merge lower planar CXZB G-code and upper conic CXZB G-code into a single file.

    Both inputs are already in CXZB 4-axis format (converted by reform.py).
    """
    os.makedirs(OUTPUT_GCODE_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_GCODE_DIR, f"{model_name}_merged.gcode")

    with open(output_path, 'w') as out:
        write_header(out)

        # --- Lower planar section (CXZB with B=0) ---
        out.write("\n; === LOWER PLANAR SECTION ===\n")
        if planar_cxzb_path and os.path.exists(planar_cxzb_path):
            with open(planar_cxzb_path, 'r') as fh:
                for line in fh:
                    out.write(line)
        else:
            out.write("; WARNING: No planar G-code found, skipping lower section\n")

        # --- Upper conic section (CXZB with B=cone_angle) ---
        out.write("\n; === UPPER CONIC SECTION ===\n")
        if conic_gcode_path and os.path.exists(conic_gcode_path):
            with open(conic_gcode_path, 'r') as fh:
                for line in fh:
                    out.write(line)
        else:
            out.write("; WARNING: No conic G-code found, skipping upper section\n")

        write_footer(out)

    print(f"Merged G-code written to {output_path}")
    return output_path
