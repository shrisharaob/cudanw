SRC = mysolver.cu

a.out : mysolver.cu
	nvcc -arch=sm_35 -O3 $<

debug : mysolver.cu
	nvcc  -arch=sm_35 -g -G -lineinfo $<

clean:
	-rm *.o *.out

.PHONY: clean