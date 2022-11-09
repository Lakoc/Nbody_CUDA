#!/bin/bash

SIZE_1=1024
N_START="N=2048"

cd /home/lakoc/PCG_NBODY/CPU
for i in $(seq 1 10); do
  N=$((2 ** i * SIZE_1))
  sed -i "/${N_START}/c\\N=${N}" Makefile
  make
  make run
  N_START="N=${N}"
done

sed -i "/${N_START}/c\\N=2048" Makefile