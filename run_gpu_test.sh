#!/bin/bash
#PBS -q qgpu
#PBS -A DD-22-68
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -N PCG-NBODY

STEP=step4
PROJECT_DIR=/home/lakoc/PCG_NBODY

ml HDF5/1.12.2-iimpi-2022a
ml CUDA/11.7.0

cd $PROJECT_DIR/$STEP
make
bash evaluate_gpu_runtime.sh
