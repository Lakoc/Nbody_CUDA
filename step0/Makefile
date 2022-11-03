# Paralelní programování na GPU (PCG 2022)
# Projekt c. 1 (cuda)
# Login: xpolok03


N=4096
DT=0.01f
STEPS=500
THREADS_PER_BLOCK=128
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=20

SAMPLE_INPUT=../sampledata/sampleInput.h5
SAMPLE_OUTPUT=../sampledata/sampleOutput.h5
OUTPUT=step0Output.h5

INCLUDE=../commons
LIBS=-lhdf5
LOCAL_LINKS=-I/usr/include/hdf5/serial/ -L/usr/lib/x86_64-linux-gnu/

FLAGS=  

.PHONY: all clean run profile

all: nbody

nbody: nbody.cu main.cu nbody.h
	nvcc ${FLAGS} -I${INCLUDE} nbody.cu main.cu ../commons/h5Helper.cpp ${LIBS} -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)

check_output:
	./nbody 4096 0.01f 2500 ${THREADS_PER_BLOCK} 50 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_y_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_z_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /weight_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w_final
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_y
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /pos_z
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_y
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /vel_z
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /weight
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_x
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_y
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_z
	-h5diff -v2 -p 0.00001 $(SAMPLE_OUTPUT) $(OUTPUT) /com_w

check_com:
	./nbody 10 0.01f 10000 ${THREADS_PER_BLOCK} 0 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)


PROFILE=/apps/all/CUDA/11.7.0/nsight-compute-2022.2.0/target/linux-desktop-glibc_2_11_3-x64/ncu --export profile_out_test \
		--force-overwrite \
		--target-processes application-only \
		--replay-mode kernel \
		--kernel-name-base function \
		--launch-skip-before-match 0 \
		--section ComputeWorkloadAnalysis \
		--section InstructionStats \
		--section LaunchStats \
		--section MemoryWorkloadAnalysis \
		--section MemoryWorkloadAnalysis_Chart \
		--section MemoryWorkloadAnalysis_Tables \
		--section Occupancy \
		--section SchedulerStats \
		--section SourceCounters \
		--section SpeedOfLight \
		--section SpeedOfLight_RooflineChart \
		--section WarpStateStats \
		--sampling-interval auto \
		--sampling-max-passes 5 \
		--sampling-buffer-size 33554432 \
		--profile-from-start 1 \
		--cache-control all \
		--clock-control base \
		--apply-rules yes \
		--check-exit-code yes

profile:
	$(PROFILE) ./nbody 56320 ${DT} 1 ${THREADS_PER_BLOCK} 0 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(SAMPLE_INPUT) $(OUTPUT)

