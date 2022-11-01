# Paralelní programování na GPU (PCG 2021)
# Projekt c. 1 (cuda)
# Login: xlogin00

LIBS=-lhdf5
LOCAL_LINKS=-I/usr/include/hdf5/serial/ -L/usr/lib/x86_64-linux-gnu/
.PHONY: all clean

all: gen

gen: gen.cpp h5Helper.cpp
	g++ ${LOCAL_LINKS} -o gen gen.cpp h5Helper.cpp ${LIBS}

clean:
	rm -f *.o nbody
