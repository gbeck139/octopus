import os


def generate_prusa_config(config, base_ini_path, output_ini_path):

    with open(base_ini_path, "r") as f:
        lines = f.readlines()

    settings_map = {
        "layer_height": config["print"]["layer_height"],
        "temperature": config["print"]["nozzle_temperature"],
        "bed_temperature": config["print"]["bed_temperature"],
        "fill_density": f'{config["print"]["infill_density"]}%'
    }

    updated_lines = []

    for line in lines:

        stripped = line.strip()

        updated = False

        for key, value in settings_map.items():

            if stripped.startswith(f"{key} ="):

                updated_lines.append(f"{key} = {value}\n")

                updated = True
                break

        if not updated:
            updated_lines.append(line)

    os.makedirs(os.path.dirname(output_ini_path), exist_ok=True)

    with open(output_ini_path, "w") as f:
        f.writelines(updated_lines)

    return output_ini_path