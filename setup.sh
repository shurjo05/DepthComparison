#!/bin/bash
# Setup script for depth model comparison
# Run this on the desktop with RTX 4060

set -e

echo "=== Installing pip dependencies ==="
pip install -r requirements.txt

echo "=== Cloning model repos ==="
# Pixel-Perfect Depth
if [ ! -d "pixel-perfect-depth" ]; then
    git clone https://github.com/gangweix/pixel-perfect-depth
    pip install -r pixel-perfect-depth/requirements.txt
fi

# VGGT (Meta)
if [ ! -d "vggt" ]; then
    git clone https://github.com/facebookresearch/vggt
    pip install -r vggt/requirements.txt
fi

# Depth Anything V3
if [ ! -d "Depth-Anything-3" ]; then
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3
    cd Depth-Anything-3 && pip install -e . && cd ..
fi

echo "=== Done! Now run: ==="
echo "  python depth_comparison/select_samples.py"
echo "  python depth_comparison/run_all_models.py"
echo "  python depth_comparison/evaluate.py"
