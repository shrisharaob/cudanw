SRC = mysolver.cu

a.out : mysolver.cu
	nvcc -arch=sm_35 -O4 $< -o nw.out
        gcc -O3 -Ofast GenerateFFConMatOnCircPatch.c -lgsl -lgslcblas -lm -o genffmat.out
debug : mysolver.cu
	nvcc  -arch=sm_35 -g -G -lineinfo $<

clean:
	-rm *.o *.out

.PHONY: clean
