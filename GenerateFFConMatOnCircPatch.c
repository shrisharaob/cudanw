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
    preFactor =  1.0 * (CFF * K) / preFactor; // FACTOR OF TWO BECAUSE BOTH POPULATIONS ARE CONSIDERED
    for(i = 0; i < NFF; ++i) {
      conProbMat[i + j * NFF] *= preFactor;
    }
    preFactor = 0.0;
  }
}

void ConProbPreFactorRec(double *convec) {
  /*  COMPUTE PRE-FACTOR AND MULTIPLY zB[clm] = K / sum(conProd(:, clm)) */
  unsigned long int i, j;
  double preFactorE2All, preFactorI2All;
  preFactorI2All = 0.0;
  preFactorE2All = 0.0;
  for(i = 0; i < N_NEURONS; ++i) { // sum over rows
    for(j = 0; j < N_NEURONS; ++j) {
      if(i < NE) {
	preFactorE2All += convec[i + j * N_NEURONS];
      }
      else {
	preFactorI2All += convec[i + j * N_NEURONS];
      }
    }
    preFactorI2All = 2.0 * (double)K / preFactorI2All;
    preFactorE2All = 2.0 * (double)K / preFactorE2All;
    for(j = 0; j < N_NEURONS; ++j) {
      if(i < NE) {
        convec[i + j * N_NEURONS] *= preFactorE2All;
      }
      else {
        convec[i + j * N_NEURONS] *= preFactorI2All;
      }
    }
    preFactorI2All = 0.0;
    preFactorE2All = 0.0;
  }
  
  /* preFactorI2All = (double)K / preFactorI2All; */
  /* preFactorE2All = (double)K / preFactorE2All; */
  /* /\* now multiply the prefactor *\/ */
  /* for(i = 0; i < N_NEURONS; ++i) { */
  /*   for(j = 0; j < N_NEURONS; ++j) {     */
  /*     if(i < NE) { */
  /*       convec[i + j * N_NEURONS] *= preFactorE2All; */
  /*     } */
  /*     else { */
  /*       convec[i + j * N_NEURONS] *= preFactorI2All; */
  /*     } */
  /*   } */
  /* } */

}


void GetXYAngle(double *xCords, double *yCords, float *angles, unsigned long nCordinates) {
  unsigned long i;
  for(i = 0; i < nCordinates; ++i) {
    angles[i] = 0.5 * (atan2(yCords[i] - PATCH_CENTER_Y, xCords[i] - PATCH_CENTER_X)) + PI * 0.5;
    //    printf("%f %f %f\n", yCords[i] - PATCH_CENTER_Y, xCords[i] - PATCH_CENTER_X, angles[i]);
  }
}

void GenXYCordinate(double *ffXcord, double *ffYcord, unsigned long nCordinates) {
  unsigned long i;
  const gsl_rng_type * TXY;
  gsl_rng *gslRGNStateXY;
  int IS_VALID;
  double xCord, yCord;
  //  gsl_rng_env_setup();
  TXY = gsl_rng_default;
  gslRGNStateXY = gsl_rng_alloc(TXY);
  gsl_rng_set(gslRGNStateXY, time(NULL));
  for(i = 0; i < nCordinates; ++i) {
    IS_VALID = 0;
    while(!IS_VALID) {
      xCord = gsl_rng_uniform(gslRGNStateXY) - PATCH_CENTER_X;
      yCord = gsl_rng_uniform(gslRGNStateXY) - PATCH_CENTER_Y;
      if((xCord * xCord) + (yCord * yCord) <= PATCH_RADIUS_SQRD) {
	ffXcord[i] = xCord + PATCH_CENTER_X;
	ffYcord[i] = yCord + PATCH_CENTER_Y;
	IS_VALID = 1;
      }
    }
  }
  gsl_rng_free(gslRGNStateXY);  
}

double Gaussian1D(double x, double xMean, double xSigma) {
  return (1.0 / sqrt(2.0 * PI)) * exp(-1 * ((x - xMean) * (x - xMean)) / (2.0 * xSigma * xSigma));
}

double ConProb(double xCord0, double xCord1, double xSigma, double yCord0, double yCord1, double ySigma) {
  double boundryX0, boundryX1, boundryY0, boundryY1;
  double fx, fy;
  boundryX0 = PATCH_RADIUS - sqrt(PATCH_RADIUS_SQRD - pow(yCord0 - PATCH_CENTER_Y, 2));
  boundryX1 = PATCH_RADIUS + sqrt(PATCH_RADIUS_SQRD - pow(yCord0 - PATCH_CENTER_Y, 2));  
  fx = Gaussian1D(xCord1, xCord0, xSigma) + Gaussian1D(xCord1, 2 * boundryX0 - xCord0, xSigma) + Gaussian1D(xCord1, 2 * boundryX1 - xCord0, xSigma);
  boundryY0 = PATCH_RADIUS - sqrt(PATCH_RADIUS_SQRD - pow(xCord0 - PATCH_CENTER_X, 2));
  boundryY1 = PATCH_RADIUS + sqrt(PATCH_RADIUS_SQRD - pow(xCord0 - PATCH_CENTER_X, 2));
  fy = Gaussian1D(yCord1, yCord0, ySigma) + Gaussian1D(yCord1, 2 * boundryY0 - yCord0, ySigma) + Gaussian1D(yCord1, 2 * boundryY1 - yCord0, ySigma);
  //  printf("%f %f %f %f", xCord0, xCord1, yCord0, yCord1);
  /* printf("%f %f %f    ", PATCH_RADIUS_SQRD, yCord0, PATCH_CENTER_Y); */
  /* printf("%f %f %f %f\n", boundryX0, boundryX1, boundryY0, boundryY1); */
  /* printf("%f %f\n", fx, fy); */
  return fx * fy;
}

int main (void)
{
  const gsl_rng_type * T;
  gsl_rng *gslRGNState;
  double *conMatFF = NULL, xa, ya;
  double *xCord = NULL, *yCord = NULL, *xCordFF = NULL, *yCordFF = NULL;
  float *xyAnglesFF = NULL, *xyAngles = NULL;
  int i, j, n = 10;
  int IF_PERIODIC = 0, IF_PRINT_CM_TO_FILE = 0;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  gslRGNState = gsl_rng_alloc(T);
  gsl_rng_set(gslRGNState, time(NULL));
  //  double u = gsl_rng_uniform (r);
  xCordFF = (double *)malloc((unsigned long long)NFF * sizeof(double));
  yCordFF = (double *)malloc((unsigned long long)NFF * sizeof(double));
  xyAnglesFF = (float *)malloc((unsigned long long)NFF * sizeof(float));  
  xCord = (double *)malloc((unsigned long long)N_NEURONS * sizeof(double));
  yCord = (double *)malloc((unsigned long long)N_NEURONS * sizeof(double));
  xyAngles = (float *)malloc((unsigned long long)N_NEURONS * sizeof(float));
  GenXYCordinate(xCordFF, yCordFF, NFF);
  GenXYCordinate(xCord, yCord, N_NEURONS);

  GetXYAngle(xCord, yCord, xyAngles, N_NEURONS);   
  GetXYAngle(xCordFF, yCordFF, xyAnglesFF, NFF);

  conMatFF = (double *)malloc((unsigned long long)NFF * N_NEURONS * sizeof(double));
  printf("\n Generating feed forward connection matrix ... "); fflush(stdout);
  for(i = 0; i < N_NEURONS; i++) {
    for(j = 0; j < NFF; j++) {
      conMatFF[i + j * N_NEURONS] = ConProb(xCord[i], xCordFF[j], FF_CON_SIGMA_X, yCord[i], yCordFF[j], FF_CON_SIGMA_Y);
        /* conMatFF[i + j * N_NEURONS] = ConProb(0.1, 0.5, FF_CON_SIGMA_X, 0.5, 0, FF_CON_SIGMA_Y);       */

    }
  }


  /* for(i = 0; i < N_NEURONS; i++) { */
  /*   for(j = 0; j < NFF; j++) { */
  /*         printf("%f \n", conMatFF[i + j * N_NEURONS]); */

  /*   } */
  /* } */
  
  printf("done\n");  
  /* TRANSPOSE TO ACUTALLY MAKE IT PROJECTIONS FROM NFF TO L 2/3 */
  printf("\n transposing matrix ... "); fflush(stdout);
  MatTranspose(conMatFF, N_NEURONS, NFF);
  printf("done\n");  
  /* MUTIPLY WITH PREFACTOR */
  printf("\n computing prefactors ... "); fflush(stdout);
  ConProbPreFactor(conMatFF);
  printf("done\n");


  
  if(IF_PRINT_CM_TO_FILE) {
    FILE *fpconmat = fopen("ffcm.csv", "w");
    for(i = 0; i < NFF; ++i) {
      for(j = 0; j < N_NEURONS; ++j) {
	fprintf(fpconmat, "%f ", conMatFF[i + j * NFF]);
      }
      fprintf(fpconmat, "\n");
    }
    fclose(fpconmat);
  }
  /* GENERATE CONNECTIVITY MATRIX */
  for(i = 0; i < NFF; ++i) {
    for(j = 0; j < N_NEURONS; ++j) {
      if(conMatFF[i + j * NFF] > gsl_rng_uniform(gslRGNState)) {
        conMatFF[i + j * NFF] = 1.0;
      }
      else {
        conMatFF[i + j * NFF] = 0.0;
      }
    }
  }
  /* GENERATE SPARSE REPRESENTATIONS */
  int idxVecFF[NFF], nPostNeuronsFF[NFF];
  int *sparseConVecFF;
  double kff = CFF * K;
  sparseConVecFF = (int *)malloc((unsigned long long)N_NEURONS * (2ULL * (unsigned long long)kff + NFF) * sizeof(int)); 
  memset(sparseConVecFF, 0, (unsigned long long)N_NEURONS * (2ULL * (unsigned long long)kff + NFF));
  printf("generating sparse representation ..."); fflush(stdout);
  GenSparseMat(conMatFF, NFF, N_NEURONS, sparseConVecFF, idxVecFF, nPostNeuronsFF);
  /* WRITE SPARSEVEC TO BINARY FILE */
  FILE *fpSparseConVecFF = NULL, *fpIdxVecFF = NULL, *fpNpostNeuronsFF = NULL;
  FILE *fpXcordsFF = NULL, *fpYcordsFF = NULL, *fpAnglesFF = NULL;
  FILE *fpXcords = NULL, *fpYcords = NULL, *fpAngles = NULL;
  /* FILE *testFloat; */
  /* testFloat = fopen("tstfloat.dat", "wb"); */
  /* float tstfltrray[3] = {0.1, 0.93, 1.8098}; */
  /* fwrite(tstfltrray, sizeof(float), 3, testFloat); */
  /* fclose(testFloat); */
  
  fpSparseConVecFF = fopen("sparseConVecFF.dat", "wb");
  unsigned int nElementsWritten, nConnections = 0;
  for(i = 0; i < NFF; ++i) {
    nConnections += nPostNeuronsFF[i];
  }
  printf("#connections = %d", nConnections);
  printf("\n avg connections = %f \n", (double)nConnections / NFF);  
  printf("done\n writing to file ... "); fflush(stdout);
  nElementsWritten = fwrite(sparseConVecFF, sizeof(int), nConnections, fpSparseConVecFF);
  fclose(fpSparseConVecFF);
  printf("\nsparseconvec: #n= %d\n", nElementsWritten);
  fpIdxVecFF = fopen("idxVecFF.dat", "wb");
  fwrite(idxVecFF, sizeof(int), NFF,  fpIdxVecFF);
  fclose(fpIdxVecFF);
  fpNpostNeuronsFF = fopen("nPostNeuronsFF.dat", "wb");
  fwrite(nPostNeuronsFF, sizeof(int), NFF, fpNpostNeuronsFF);
  fclose(fpNpostNeuronsFF);
  
  fpXcordsFF = fopen("xCordsFF.dat", "wb");
  fwrite(xCordFF, sizeof(double), NFF, fpXcordsFF);
  fclose(fpXcordsFF);
  fpYcordsFF = fopen("yCordsFF.dat", "wb");
  fwrite(yCordFF, sizeof(double), NFF, fpYcordsFF);
  fclose(fpYcordsFF);
  fpAnglesFF = fopen("poAnglesFF.dat", "wb");
  fwrite(xyAnglesFF, sizeof(float), NFF, fpAnglesFF);
  fclose(fpAnglesFF);

  fpXcords = fopen("xCords.dat", "wb");
  fwrite(xCord, sizeof(double), N_NEURONS, fpXcords);
  fclose(fpXcords);
  fpYcords = fopen("yCords.dat", "wb");
  fwrite(yCord, sizeof(double), N_NEURONS, fpYcords);
  fclose(fpYcords);
  fpAngles = fopen("poAngles.dat", "wb");
  fwrite(xyAngles, sizeof(float), N_NEURONS, fpAngles);
  fclose(fpAngles);
    
  printf("done\n");
  free(conMatFF);
  free(sparseConVecFF);
  free(xyAngles);
  free(xyAnglesFF);
  
  // GENEREATE RECURRENT CONNECTIONS
  double *conMat = NULL;
  conMat = (double *)malloc((unsigned long long)N_NEURONS * N_NEURONS * sizeof(double)); // RECURRENT
  printf("\n Generating recurrent connection matrix ... "); fflush(stdout);
  for(i = 0; i < N_NEURONS; i++) {
    for(j = 0; j < N_NEURONS; j++) {
           conMat[i + j * N_NEURONS] = ConProb(xCord[i], xCord[j], CON_SIGMA_X, yCord[i], yCord[j], CON_SIGMA_Y);      // i to j
      /* conMat[i + j * N_NEURONS] = ConProb(0.1, 0.5, CON_SIGMA_X, 0.5, 0.0, CON_SIGMA_Y);      // i to j */
      /* printf("%f \n", conMat[i + j * N_NEURONS]);    */
    }
  }
  printf("done\n");  
  printf("\n computing prefactors ... "); fflush(stdout);
  ConProbPreFactorRec(conMat);
  printf("done\n");

  /* double preFactorI2All = 0.0,  preFactorE2All = 0.0; */
  /* for(i = 0; i < N_NEURONS; ++i) { // sum over rows */
  /*   for(j = 0; j < N_NEURONS; ++j) { */
  
  /*     if(i < NE) { */
  /* 	preFactorE2All += conMat[i + j * N_NEURONS]; */
  /*     } */
  /*     else { */
  /* 	preFactorI2All += conMat[i + j * N_NEURONS]; */
  /*     } */
  /*   } */
  /* } */
  /* printf("\n\n ===>%f %f \n", preFactorI2All, preFactorE2All); */


  
  /* GENERATE CONNECTIVITY MATRIX */
  for(i = 0; i < N_NEURONS; ++i) {
    for(j = 0; j < N_NEURONS; ++j) {
      if(conMat[i + j * N_NEURONS] >= gsl_rng_uniform(gslRGNState)) {
        conMat[i + j * N_NEURONS] = 1.0;
      }
      else {
        conMat[i + j * N_NEURONS] = 0.0;
      }
    }
  }
  //
  nConnections = 0;  
  for(i = 0; i < N_NEURONS; ++i) {
    for(j = 0; j < N_NEURONS; ++j) {
      nConnections += conMat[i + j * N_NEURONS];
    }
  }
  printf("\n \n'nxxxxxxxxx #edges: %u \n\n ", nConnections);
  /* GENERATE SPARSE REPRESENTATIONS */
  int *idxVec, *nPostNeurons, *sparseConVec;
  idxVec = (int *)malloc((unsigned long long)N_NEURONS * sizeof(int));
  nPostNeurons = (int *)malloc((unsigned long long)N_NEURONS * sizeof(int));
  sparseConVec = (int *)malloc((unsigned long long)N_NEURONS * (2ULL * (unsigned long long)K + N_NEURONS) * sizeof(int));   
  memset(sparseConVec, 0, (unsigned long long)N_NEURONS * (2ULL * (unsigned long long)K + N_NEURONS));
  printf("generating sparse representation ..."); fflush(stdout);
  GenSparseMat(conMat, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons);
  /* WRITE SPARSEVEC TO BINARY FILE */
  FILE *fpSparseConVec = NULL, *fpIdxVec = NULL, *fpNpostNeurons = NULL;
  fpSparseConVec = fopen("sparseConVec.dat", "wb");
  nElementsWritten = 0;
  nConnections = 0;  
  for(i = 0; i < N_NEURONS; ++i) {
    nConnections += nPostNeurons[i];
  }
  printf("#connections = %d\n", nConnections);
  printf("done\n writing to file ... "); fflush(stdout);
  nElementsWritten = fwrite(sparseConVec, sizeof(int), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  printf("\nsparseconvec: #n= %d\n", nElementsWritten);
  fpIdxVec = fopen("idxVec.dat", "wb");
  fwrite(idxVec, sizeof(int), N_NEURONS,  fpIdxVec);
  fclose(fpIdxVec);
  fpNpostNeurons = fopen("nPostNeurons.dat", "wb");
  fwrite(nPostNeurons, sizeof(int), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  printf("done\n");


  FILE *fp, *fp01;
  fp01 = fopen("countI.csv", "w");
  fp = fopen("countE.csv", "w");
  int countE = 0, countI = 0;
  for(i = 0; i < N_NEURONS; ++i) {
    countI = 0;
    countE = 0;
    for(j = 0; j < N_NEURONS; ++j) {
      if(j < NE) {
        countE += conMat[i * N_NEURONS + j];   
      }
      else {
        countI += conMat[i * N_NEURONS + j];   
      }
    }

    fprintf(fp, "%d\n", countE); 
    fprintf(fp01, "%d\n", countI);
  }
  printf("cntE, cntI  %d %d\n", countE, countI);  
  fclose(fp);   
  fclose(fp01);

  

  free(conMat);
  free(idxVec);
  free(nPostNeurons);
  free(sparseConVec);
  
  free(xCord);
  free(yCord);
  free(xCordFF);
  free(yCordFF);  
  gsl_rng_free(gslRGNState);
  return 0;
}
