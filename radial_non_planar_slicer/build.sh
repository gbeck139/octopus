#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Path to python executable in venv
PYTHON_ENV="venv/bin/python"
PYINSTALLER="venv/bin/pyinstaller"

# Check if venv exists
if [ ! -f "$PYINSTALLER" ]; then
    echo "Error: PyInstaller not found in venv at $PYINSTALLER"
    exit 1
fi

echo "Cleaning previous build..."
rm -rf build dist

echo "Running PyInstaller..."
# Run PyInstaller
"$PYINSTALLER" radial_slicer.spec --clean --noconfirm

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "PyInstaller failed."
    exit 1
fi

echo "Copying input models..."
cp -r input_models dist/

echo "Build complete. Executable is ready in dist/"
