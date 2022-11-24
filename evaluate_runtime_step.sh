#!/bin/bash
SIZE_1=5
SIZE_2=512
DT=0.01f
STEPS=1
THREADS_PER_BLOCK=512
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=0

SAMPLE_INPUT=sampledata/input_long.h5
SAMPLE_OUTPUT=sampledata/input_long_out.h5

STEP=$1


for i in $(seq 10 25); do
  N=$((SIZE_1 * i * SIZE_2))
  ${STEP}/nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} ${SAMPLE_INPUT} ${SAMPLE_OUTPUT} >> ${STEP}/runtime.txt
  echo "" >> ${STEP}/runtime.txt
done
