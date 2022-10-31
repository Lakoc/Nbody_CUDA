#!/bin/bash
#PBS -q qgpu
#PBS -A DD-22-68
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -N PCG-NBODY

STEP=step0
PROJECT_DIR=/home/lakoc/PCG_NBODY

ml HDF5/1.12.2-iimpi-2022a
ml CUDA/11.7.0
ml Python/3.10.4-GCCcore-11.3.0

cd $PROJECT_DIR
cd $STEP
make
cd ../tests
python3 -m venv py-test-env

source py-test-env/bin/activate
python3 -m pip install h5py
run_tests.sh $STEP/nbody
deactivate