SRC = mysolver.cu

a.out : mysolver.cu
	nvcc -g -G -arch=sm_11 $<

clean:
	-rm *.o *.out

.PHONY: clean