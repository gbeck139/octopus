import json
import copy

from .defaults import DEFAULT_CONFIG


def deep_merge(defaults, user_config):
    """
    Recursively merge user config into defaults.
    """

    result = copy.deepcopy(defaults)

    for key, value in user_config.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)

        else:
            result[key] = value

    return result


def load_config(config_path):
    """
    Load JSON config and merge with defaults.
    """

    with open(config_path, "r") as f:
        user_config = json.load(f)

    merged_config = deep_merge(DEFAULT_CONFIG, user_config)

    return merged_config