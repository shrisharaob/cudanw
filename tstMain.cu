#include <stdio.h>
#include <cuda.h>
#include "GenSparseMat.cu"

int main() {
  int conVec[] = {0, 1, 1, 
                  0, 0, 0,
                  1, 0, 1};
  int rows = 3, clms = 3;
  int sparseVec[rows * rows], idxVec[rows], nPostNeurons[rows];
  
  GenSparseMat(conVec, rows, clms, sparseVec, idxVec, nPostNeurons);
  
  printf("conVec :  ");
  for(int i = 0; i < rows * rows; ++i) {
    printf("%d ", conVec[i]);
  }
  printf("\n");

  printf("nPostNeurons :  ");
  for(int i = 0; i < rows; ++i) {
    printf("%d ", nPostNeurons[i]);
  }
  printf("\n");  

  printf("idxVec :  ");
  for(int i = 0; i < rows; ++i) {
    printf("%d ", idxVec[i]);
  }
  printf("\n");  
  
  int kk = 0;

  for(int i = 0; i < rows; ++i) {
    kk += nPostNeurons[i];
  }

  printf("sparseVec :  ");
  for(int i = 0; i < kk; ++i) {
    printf("%d ", sparseVec[i]);
  }
  printf("\n");
}
  
