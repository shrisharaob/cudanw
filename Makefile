SRC = mysolver.cu

a.out : mysolver.cu
	nvcc -arch=sm_11 -O3 $<

debug : mysolver.cu
	nvcc -g -G -lineinfo -arch=sm_11 $<

clean:
	-rm *.o *.out

.PHONY: clean