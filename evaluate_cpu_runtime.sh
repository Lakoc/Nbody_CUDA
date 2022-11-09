#!/bin/bash

PROJECT_DIR="/home/lakoc/PCG_NBODY"

CPU_DIR="${PROJECT_DIR}/${CPU}"

SIZE_1=1024
N_START="N=2048"

cd $CPU_DIR
for i in $(seq 1 10); do
  N=$((2 ** i * SIZE_1))
  sed -i "/${N_START}/c\\N=${N}" Makefile
  make
  make run
  N_START="N=${N}"
done

