#!/bin/bash

# Build script for creating distributable radial slicer package
# This script will:
# 1. Install dependencies
# 2. Run PyInstaller to bundle the Python application
# 3. Copy PrusaSlicer to the dist folder
# 4. Create a complete distributable package

set -e  # Exit on error

echo "========================================="
echo "Radial Slicer Build Script"
echo "========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install dependencies
echo ""
echo "Step 1: Installing Python dependencies..."
pip3 install -r requirements.txt

# Run PyInstaller
echo ""
echo "Step 2: Building executable with PyInstaller..."
pyinstaller radial_slicer.spec --clean

# Create PrusaSlicer directory in dist
echo ""
echo "Step 3: Setting up PrusaSlicer in distribution..."
mkdir -p dist/PrusaSlicer

# Instructions for PrusaSlicer
echo ""
echo "========================================="
echo "PrusaSlicer Setup Instructions"
echo "========================================="
echo ""
echo "To complete the distribution, you need to add PrusaSlicer:"
echo ""
echo "Option 1 (Linux AppImage):"
echo "  1. Download PrusaSlicer AppImage from https://www.prusa3d.com/page/prusaslicer_424/"
echo "  2. Copy it to: dist/PrusaSlicer/PrusaSlicer.AppImage"
echo "  3. Make it executable: chmod +x dist/PrusaSlicer/PrusaSlicer.AppImage"
echo ""
echo "Option 2 (Windows):"
echo "  1. Download PrusaSlicer installer from https://www.prusa3d.com/page/prusaslicer_424/"
echo "  2. Install PrusaSlicer"
echo "  3. Copy prusa-slicer.exe (and dependencies) to: dist/PrusaSlicer/"
echo ""
echo "Option 3 (macOS):"
echo "  1. Download PrusaSlicer DMG from https://www.prusa3d.com/page/prusaslicer_424/"
echo "  2. Copy the app bundle to: dist/PrusaSlicer/"
echo ""

# Create output directories
echo ""
echo "Step 4: Creating output directories..."
mkdir -p dist/radial_non_planar_slicer/input_gcode
mkdir -p dist/radial_non_planar_slicer/output_gcode
mkdir -p dist/radial_non_planar_slicer/output_models

# Copy config files
echo ""
echo "Step 5: Copying configuration files..."
cp -r radial_non_planar_slicer/prusa_slicer dist/radial_non_planar_slicer/
cp -r radial_non_planar_slicer/input_models dist/radial_non_planar_slicer/ 2>/dev/null || echo "Warning: No input models found to copy"

echo ""
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Your distributable package is in the 'dist' folder."
echo "Don't forget to add PrusaSlicer as described above."
echo ""
echo "To distribute:"
echo "  1. Add PrusaSlicer to dist/PrusaSlicer/"
echo "  2. Zip/tar the entire 'dist' folder"
echo "  3. Users can extract and run 'radial_slicer' directly"
echo ""
