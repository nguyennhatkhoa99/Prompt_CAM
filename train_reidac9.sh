#!/bin/bash

# Training script for Reidac9 dataset with Prompt_CAM
#
# Usage:
#   ./train_reidac9.sh              # Train with DINO (default)
#   ./train_reidac9.sh dino         # Train with DINO backbone
#   ./train_reidac9.sh dinov2       # Train with DINOv2 backbone

BACKBONE=${1:-dino}

echo "=============================================="
echo "Training Prompt_CAM on Reidac9 Dataset"
echo "Backbone: $BACKBONE"
echo "=============================================="

if [ "$BACKBONE" = "dino" ]; then
    CONFIG="experiment/config/prompt_cam/dino/reidac9/args.yaml"
elif [ "$BACKBONE" = "dinov2" ]; then
    CONFIG="experiment/config/prompt_cam/dinov2/reidac9/args.yaml"
else
    echo "Error: Unknown backbone '$BACKBONE'. Use 'dino' or 'dinov2'."
    exit 1
fi

echo "Config: $CONFIG"
echo ""

python main.py --config "$CONFIG"

echo ""
echo "Training completed!"
echo "Check output directory for results and checkpoints."
