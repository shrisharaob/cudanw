SRC = mysolver.c

mysolver: aux.o $(SRC)
##	gcc -o mysolver $(SRC) cudaAuxFuncs.o cudakernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lm 
	gcc -o mysolver $(SRC) aux.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand -lm -O3 -ftree-vectorizer-verbose=1
##	gcc -o mysolver $(SRC) cudaAuxFuncs.o cudakernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand -lm -g
cudakernel.o: cudakernel.cu
	nvcc -arch sm_11 --use_fast_math -O3 -I/usr/local/cuda/include -c -o aux.o aux.cu

clean:
	-rm *.o mysolver

.PHONY: clean