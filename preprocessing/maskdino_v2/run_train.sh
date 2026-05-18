#!/usr/bin/env bash
# Full MaskDINO v2 training: 10K iters at IMS_PER_BATCH=4, ~6h on H200.
set -eo pipefail

source /etc/profile.d/modules.sh
module purge
module load Miniforge3
export PYTHONNOUSERSITE=1
conda activate swine-rgbd
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

PIGREL=/mnt/gs21/scratch/basharmk/data/unl/pigformer_release
MASKDINO=/mnt/gs21/scratch/basharmk/data/unl/MaskDINO
export PIG_DEPTH_V2_DATASET_ROOT=$MASKDINO/datasets/pig_depth_combined_v2

cd "$MASKDINO"
python train_net.py \
    --config-file "$PIGREL/preprocessing/maskdino_v2/maskdino_R50_depth_v2.yaml" \
    --num-gpus 1 \
    OUTPUT_DIR "$MASKDINO/output/pig_depth_v2_endpoint" \
    MODEL.WEIGHTS "$MASKDINO/weights/r50_conv1_1ch.pkl"
