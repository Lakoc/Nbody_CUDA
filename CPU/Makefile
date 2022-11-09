# Architektura procesoru (ACH 2016)
# Projekt c. 1 (nbody)
# Login: xlogin00

CC=icpc
CFLAGS=-std=c++11 -lpapi -ansi-alias
OPT=-O2 -Wall
#REPORT=-qopt-report=5

N=2048
DT=0.01f
STEPS=1

PARAMS=-DN=$(N) -DDT=$(DT) -DSTEPS=$(STEPS)

INPUT=../sampledata/sampleInput.h5
OUTPUT=out.h5

PAPI_EVENTS=PAPI_FP_OPS|PAPI_SP_OPS

all:
	$(CC) $(CFLAGS) $(OPT) $(REPORT) $(PARAMS) -c velocity.cpp
	$(CC) $(CFLAGS) $(OPT) $(REPORT) $(PARAMS) -c nbody.cpp
	$(CC) $(CFLAGS) $(OPT) $(PARAMS) -S -fsource-asm -c nbody.cpp
	$(CC) $(CFLAGS) $(OPT) $(PARAMS) velocity.o nbody.o main.cpp -o nbody
	$(CC) $(CFLAGS) gen.cpp -o gen

clean:
	rm -f *.o nbody gen

run:
	PAPI_EVENTS='$(PAPI_EVENTS)' ./nbody $(INPUT) $(OUTPUT)
