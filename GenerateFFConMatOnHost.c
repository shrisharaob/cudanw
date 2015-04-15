#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "hostConstants.h"

void GenSparseMat(double *conVec,  int rows, int clms, int* sparseVec, int* idxVec, int* nPostNeurons ) {
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
      if((int)conVec[i + rows * j]) { /* i --> j  */
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


void MatTranspose(double *m, int w, int h) {
  // transpose of flattened matrix 
  // m : flattened matrix  -  row + clm * linearId
  // w : # of columns
  // h : # of rows
  int start, next, i;
  double tmp;
  for (start = 0; start <= w * h - 1; start++) {
    next = start;
    i = 0;
    do {i++;
      next = (next % h) * w + next / h;
    } while (next > start);
    if (next < start || i == 1) continue;
    tmp = m[next = start];
    do {
      i = (next % h) * w + next / h;
      m[next] = (i == start) ? tmp : m[i];
      next = i;
    } while (next > start);
  }
}

void ConProbPreFactor(double *conProbMat) {
  double preFactor = 0.0;
  int i, j;
  for(j = 0; j < N_NEURONS; ++j) {
    for(i = 0; i < NFF; ++i) {
      preFactor += conProbMat[i + j * NFF];
    }
    preFactor = (CFF * K) / preFactor;
    for(i = 0; i < NFF; ++i) {
      conProbMat[i + j * NFF] *= preFactor;
    }
    preFactor = 0.0;
  }
}


double FF_YCordinate(unsigned long int neuronIdx) {
  // nA - number of FF neurons
  double nA = (double)NFF;
  return fmod((double)neuronIdx, sqrt(nA)) * (L / (sqrt(nA) - 1));
}

double FF_XCordinate(unsigned long  neuronIdx) {
  double nA = (double)NFF;
  return floor((double)neuronIdx / sqrt(nA)) * (L / (sqrt(nA) - 1));   
}

double YCordinate(unsigned long int neuronIdx) {
  // nA - number of E or I cells
  double nA = (double)NE;
  if(neuronIdx > NE - 1) { // since neuronIds for inhibitopry start from NE jusqu'a N_NEURONS
    neuronIdx -= NE;
    nA = (double)NI;
  }
  return fmod((double)neuronIdx, sqrt(nA)) * (L / (sqrt(nA) - 1));
}

double XCordinate(unsigned long  neuronIdx) {
  double nA = (double)NE;
  if(neuronIdx > NE - 1) {
    neuronIdx -= NE;
    nA = (double)NI;
  }
  return floor((double)neuronIdx / sqrt(nA)) * (L / (sqrt(nA) - 1));   
}

double Gaussian1D(double distance, double sigma) {
  double denom = (2.0 * sigma * sigma); 
  double z1 = (1.0 / (sigma * sqrt( 2.0 * PI)));
  double dummy = z1 * exp(-1.0 * (distance * distance) / denom);
  return  z1 * exp(-1.0 * (distance * distance) / denom);
}

double Gaussian2D(double x, double y, double varianceOfGaussian) {
  double z1 = (1.0 / sqrt(2.0 * PI * varianceOfGaussian)); // make global var
  double denom = (2.0 * varianceOfGaussian * varianceOfGaussian); // global var
  return  z1 * z1 * exp(-1 * pow(x, 2) / (denom)) * z1 * z1 * exp(-1 * pow(y, 2) / (denom));
}

double ShortestDistOnCirc(double point0, double point1, double perimeter) {
  double dist = 0.0;
  dist = abs(point0 - point1);
  dist = fmod(dist, perimeter);
  if(dist > 0.5){
    dist = 1.0 - dist;
  }
  return dist;
}

double ConProb(double xa, double ya, double xb, double yb, double patchSize, double varianceOfGaussian, int IF_PERIODIC) {
  double distX = 0.0; //ShortestDistOnCirc(xa, xb, patchSize);
  double distY = 0.0; //ShortestDistOnCirc(ya, yb, patchSize);
  double out = 0.0;
  if(IF_PERIODIC) {
    distX = ShortestDistOnCirc(xa, xb, patchSize);
    distY = ShortestDistOnCirc(ya, yb, patchSize);
  }
  else {
    distX = xa - xb;
    distY = ya - yb;
  }
  // return Gaussian2D(distX, distY, varianceOfGaussian);
  //  double dummy = Gaussian1D(distX, varianceOfGaussian) * Gaussian1D(distY, varianceOfGaussian);
  return Gaussian1D(distX, varianceOfGaussian) * Gaussian1D(distY, varianceOfGaussian);
}


int main (void)
{
  const gsl_rng_type * T;
  gsl_rng *gslRGNState;
  double *conMat = NULL, xa, ya;
  int i, j, n = 10;
  int IF_PERIODIC = 0, IF_PRINT_CM_TO_FILE = 0;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  gslRGNState = gsl_rng_alloc(T);
  gsl_rng_set(gslRGNState, time(NULL));
  //  double u = gsl_rng_uniform (r);
  
  conMat = (double *)malloc((unsigned long long)NFF * N_NEURONS * sizeof(double));
  printf("\n Generating feed forward connection matrix ... "); fflush(stdout);
  for(i = 0; i < N_NEURONS; i++) {
    xa = XCordinate(i);
    ya = YCordinate(i);
    
    for(j = 0; j < NFF; j++) {
      /* if(j == 0) { */
      /* 	printf("(%d, %d)(%llu)(%f, %f)\n", i, j, i + j * N_NEURONS, xa, ya); */
      /* } */
      conMat[i + j * N_NEURONS] = ConProb(xa, ya, FF_XCordinate(j), FF_YCordinate(j), L_FF, FF_CON_SIGMA, IF_PERIODIC);
    }
  }
  printf("done\n");  
  /* TRANSPOSE TO ACUTALLY MAKE IT PROJECTIONS FROM NFF TO L 2/3 */
  printf("\n transposing matrix ... "); fflush(stdout);
  MatTranspose(conMat, N_NEURONS, NFF);
  printf("done\n");  
  /* MUTIPLY WITH PREFACTOR */
  printf("\n computing prefactors ... "); fflush(stdout);
  ConProbPreFactor(conMat);
  printf("done\n");  
  if(IF_PRINT_CM_TO_FILE) {
    FILE *fpconmat = fopen("ffcm.csv", "w");
    for(i = 0; i < NFF; ++i) {
      for(j = 0; j < N_NEURONS; ++j) {
	fprintf(fpconmat, "%f ", conMat[i + j * NFF]);
      }
      fprintf(fpconmat, "\n");
    }
    fclose(fpconmat);
  }
  /* GENERATE CONNECTIVITY MATRIX */
  for(i = 0; i < NFF; ++i) {
    for(j = 0; j < N_NEURONS; ++j) {
      if(conMat[i + j * NFF] > gsl_rng_uniform(gslRGNState)) {
        conMat[i + j * NFF] = 1.0;
      }
      else {
        conMat[i + j * NFF] = 0.0;
      }
    }
  }

  /* GENERATE SPARSE REPRESENTATIONS */
  int idxVecFF[NFF], nPostNeuronsFF[NFF];
  int *sparseConVec;
  double kff = CFF * K;
  sparseConVec = (int *)malloc((unsigned long long)N_NEURONS * (2ULL * (unsigned long long)kff + NFF) * sizeof(int)); 
  memset(sparseConVec, 0, (unsigned long long)N_NEURONS * (2ULL * (unsigned long long)kff + NFF));
  printf("generating sparse representation ..."); fflush(stdout);
  GenSparseMat(conMat, NFF, N_NEURONS, sparseConVec, idxVecFF, nPostNeuronsFF);

  /* WRITE SPARSEVEC TO BINARY FILE */
  FILE *fpSparseConVecFF = NULL, *fpIdxVecFF = NULL, *fpNpostNeuronsFF = NULL;
  fpSparseConVecFF = fopen("sparseConVecFF.dat", "wb");
  unsigned int nElementsWritten, nConnections = 0;
  for(i = 0; i < NFF; ++i) {
    nConnections += nPostNeuronsFF[i];
  }
  printf("#connections = %d", nConnections);

  //  nElementsWritten = fwrite(sparseConVec, sizeof(int), N_NEURONS * (2 * (int)kff + NFF), fpSparseConVecFF);
  printf("done\n writing to file ... "); fflush(stdout);
  nElementsWritten = fwrite(sparseConVec, sizeof(int), nConnections, fpSparseConVecFF);
  //      fflush(fpSparseConVecFF);
  fclose(fpSparseConVecFF);
  printf("\nsparseconvec: #n= %d\n", nElementsWritten);
  fpIdxVecFF = fopen("idxVecFF.dat", "wb");
  fwrite(idxVecFF, sizeof(int), NFF,  fpIdxVecFF);
  fclose(fpIdxVecFF);
  fpNpostNeuronsFF = fopen("nPostNeuronsFF.dat", "wb");
  fwrite(nPostNeuronsFF, sizeof(int), NFF, fpNpostNeuronsFF);
  fclose(fpNpostNeuronsFF);
  printf("done\n");

  int IF_PRINT = 0;
  if(N_NEURONS < 20 & NFF < 10) {
    IF_PRINT = 1;
  }

  if(IF_PRINT) {
    printf("\nconnection matrix:\n");
    for(i = 0; i < NFF; ++i) {
      for(j = 0; j < N_NEURONS; ++j) {
        printf("%d ", (int)conMat[i + j * NFF]);
      }
      printf("\n");
    }	


    printf("\n");
    for(i = 0; i < NFF; ++i) {
      printf("neuron %d projects to : ", i);
      for(j = 0; j < nPostNeuronsFF[i]; ++j) {
        printf("%d ", sparseConVec[idxVecFF[i] + j]);
      }
      printf("\n");
    }
  }

  int countFF = 0;
  FILE *fpFFCount = fopen("ffcount.csv", "w"); 
  for(i = 0; i < N_NEURONS; i++) {
    countFF = 0;
    for(j = 0; j < NFF; j++) {
      countFF += (int)conMat[j + i * NFF];
    }
    fprintf(fpFFCount, "%d\n", countFF);
  }
  fclose(fpFFCount);
 

  free(conMat);
  free(sparseConVec);
  gsl_rng_free(gslRGNState);
  return 0;
}
