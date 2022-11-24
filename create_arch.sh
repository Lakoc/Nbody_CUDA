STEP0="step0/main.cu step0/Makefile step0/nbody.cu step0/nbody.h"
STEP1="step1/main.cu step1/Makefile step1/nbody.cu step1/nbody.h"
STEP2="step2/main.cu step2/Makefile step2/nbody.cu step2/nbody.h"
STEP3="step3/main.cu step3/Makefile step3/nbody.cu step3/nbody.h"
STEP4="step4/main.cu step4/Makefile step4/nbody.cu step4/nbody.h"
SCRIPTS="parse_runtimes.py evaluate_cpu_runtime.sh evaluate_gpu_runtime.sh evaluate_runtime_step.sh"
OUT_FILE="nbody.txt"
zip xpolok03.zip ${STEP0} ${STEP1} ${STEP2} ${STEP3} ${STEP4} ${SCRIPTS} ${OUT_FILE}