#ifndef _GENSPARSEMAT_
#define _GENSPARSEMAT_
void GenSparseMat(int *conVec,  int rows, int clms, int* sparseVec, int* idxVec, int* nPostNeurons ) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  
  int i, j, counter = 0, nPost;
  for(i = 0; i < rows; ++i) {
    nPost = 0;
    for(j = 0; j < clms; ++j) {
      if(conVec[i + clms * j]) { /* i --> j  */
        sparseVec[counter] = j;
	counter += 1;
        nPost += 1;
      }
    }
    nPostNeurons[i] = nPost;
  }
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPostNeurons[i-1];
  }
}

void GenPreSparseMat(int *conVec,  int rows, int clms, int* sparseVec, int* idxVec, int* nPreNeurons ) {
  /* generate sparse representation of Pre-synaptic neurons to a given neuron 
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  
  int i, j, counter = 0, nPre;
  for(i = 0; i < clms; ++i) {
    /*  fprintf(stdout, "neuron %d receives input from : ", i);
	fflush(stdout);*/
    nPre = 0;
    for(j = 0; j < rows; ++j) {
      if(conVec[i * clms + j]) { /* i <-- j  */
        sparseVec[counter] = j;
	/*	fprintf(stdout, "%d ", j); fflush(stdout);*/
	counter += 1;
        nPre += 1;
      }
    }
    nPreNeurons[i] = nPre;
    /*    fprintf(stdout, "nPre = %d,  \n", nPre);*/
  }
  idxVec[0] = 0;
  for(i = 1; i < rows; ++i) {
    idxVec[i] = idxVec[i-1] + nPreNeurons[i-1];
    /*    fprintf(stdout, "%d ", idxVec[i]);fflush(stdout);*/
  }
}
#endif
