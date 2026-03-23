#!/bin/bash
# Setup script — installs deps and clones repos that need local checkout
set -e

echo "=== Installing pip dependencies ==="
pip install -r requirements.txt

echo "=== Installing MoGe ==="
pip install git+https://github.com/microsoft/MoGe.git

echo "=== Cloning Pixel-Perfect Depth ==="
if [ ! -d "pixel-perfect-depth" ]; then
    git clone https://github.com/gangweix/pixel-perfect-depth
    pip install -r pixel-perfect-depth/requirements.txt
fi

echo "=== Cloning VGGT ==="
if [ ! -d "vggt" ]; then
    git clone https://github.com/facebookresearch/vggt
    pip install -r vggt/requirements.txt
fi

echo "=== Cloning Depth Anything V3 ==="
if [ ! -d "Depth-Anything-3" ]; then
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3
    pip install -e Depth-Anything-3
fi

echo "=== Cloning DepthFM ==="
if [ ! -d "depth-fm" ]; then
    git clone https://github.com/CompVis/depth-fm
    pip install -r depth-fm/requirements.txt
fi

echo ""
echo "=== Done! Run: ==="
echo "  python depth_comparison/select_samples.py"
echo "  python depth_comparison/run_all_models.py"
echo "  python depth_comparison/evaluate.py"
