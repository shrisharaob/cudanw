SRC = mysolver.cu

a.out : mysolver.cu
	nvcc -arch=sm_35 -O4 $< -o nw.out
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out
#	nvcc -arch=sm_35 -O4 GenFeedForwardConVecFile.cu -o genffconvec.out
	gcc -O3 -Ofast GenerateFFConMatOnHost.c -lgsl -lgslcblas -lm -o genffmat.out
debug : mysolver.cu
	nvcc  -arch=sm_35 -g -G -lineinfo $<

clean:
	-rm *.o *.out

.PHONY: clean