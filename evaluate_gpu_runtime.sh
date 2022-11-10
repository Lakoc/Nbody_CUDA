#!/bin/bash
SIZE_1=2
SIZE_2=1024
DT=0.01f
STEPS=1
THREADS_PER_BLOCK=(128 128 128 256 128 128 256 256 1024 512)
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=0

SAMPLE_INPUT=sampledata/cpu_test.h5
SAMPLE_OUTPUT=aaaa

STEP=$1

metrics="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"

# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics+="dram__bytes.sum.per_second"

for i in $(seq 1 10); do
  N=$((SIZE_1 ** i * SIZE_2))
  ncu --metrics ${metrics} --csv ${STEP}/nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK[$i-1]} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} ${SAMPLE_INPUT} ${SAMPLE_OUTPUT} > profile_${i}.out 2>&1
done