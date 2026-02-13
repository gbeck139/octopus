import sys
import os

def get_base_path():
    """
    Get the base path of the application.
    If frozen (PyInstaller), returns the directory of the executable.
    If running as a script, returns the directory of the script.
    """
    if getattr(sys, 'frozen', False):
        # running in a bundle
        return os.path.dirname(sys.executable)
    else:
        # running in a normal Python environment
        return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(relative_path):
    """
    Get the absolute path of a resource relative to the base path.
    """
    base_path = get_base_path()
    return os.path.join(base_path, relative_path)
