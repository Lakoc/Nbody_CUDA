#!/bin/bash
SIZE_1=2
SIZE_2=1024
DT=0.01f
STEPS=1
THREADS_PER_BLOCK=512
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=0

SAMPLE_INPUT=sampledata/cpu_test.h5
SAMPLE_OUTPUT=aaaa

STEP=$1


PROFILE="/apps/all/CUDA/11.7.0/nsight-compute-2022.2.0/target/linux-desktop-glibc_2_11_3-x64/ncu --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --check-exit-code yes"

for i in $(seq 1 10); do
  N=$((SIZE_1 ** i * SIZE_2))
  ${PROFILE} --export ${STEP}/profile_out_${i} ${STEP}/nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} ${SAMPLE_INPUT} ${SAMPLE_OUTPUT}
done