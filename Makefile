SRC = mysolver.cu

a.out : mysolver.cu
	nvcc -arch=sm_35 -O3 $< -o nw.out
	nvcc -arch=sm_35 -O2 GenerateConVecFile.cu -o genconvec.out
	g++ -std=c++0x -Ofast rewiringFuncs.cpp        
debug : mysolver.cu
	nvcc  -arch=sm_35 -g -G -lineinfo $<

clean:
	-rm *.o *.out

.PHONY: clean
