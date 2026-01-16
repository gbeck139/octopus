import subprocess

class CuraEngineWrapper:
    def __init__(self, cura_path, machine_def):
        self.cura_path = cura_path
        self.machine_def = machine_def

    def slice(self, stl_path, output_gcode, cura_settings):
        cmd = [
            self.cura_path, "slice",
            "-j", self.machine_def,
            "-l", stl_path,
            "-o", output_gcode
        ]
        # Get process settings CLI
        for key, value in cura_settings.items():
            cmd.extend(["-s", f"{key}={value}"])
        
        print("Running CuraEngine:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)
