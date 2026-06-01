# Needed to run in cluster enviroment
source /etc/profile.d/modules.sh
module purge
module load Miniforge3
export PYTHONNOUSERSITE=1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
conda activate swine-rgbd