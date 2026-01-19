import subprocess
import os

class CuraEngineWrapper:
    def __init__(self, cura_path, resources_path):
        self.cura_path = cura_path
        self.resources_path = resources_path

    def slice(self, stl_path, output_gcode, json_files, overrides=None):
        '''
        Docstring for slice
        
        :param self: Description
        :param stl_path: Description
        :param output_gcode: Description
        :param json_files: list of JSON paths (machine, material, process)
        :param overrides: dictionary of -s overrides (optional), will implement later when "edit"
        '''

        cmd = [self.cura_path, "slice"]

        # Add all JSON files in order
        for j in json_files:
            cmd.extend(["-j", j])

        # Model + output
        cmd.extend([
            "-l", stl_path,
            "-o", output_gcode
        ])

         # Optional overrides (-s only if needed)
        if overrides:
            for key, value in overrides.items():
                cmd.extend(["-s", f"{key}={value}"])
        
        #Convert paths to absolute
        #stl_path = os.path.abspath(stl_path)
        #output_gcode = os.path.abspath(output_gcode)
        #json_files = [os.path.abspath(j) for j in json_files]
            

        print("Running CuraEngine:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)
