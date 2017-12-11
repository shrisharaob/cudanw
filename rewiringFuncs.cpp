#include <stdio.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <ctime>
#include <string>
#include <cstring>

#include "devHostConstants.h"
using namespace::std ;

int *sparseConVec = NULL;

void LoadSparseConMat(int *idxVec, int *nPostNeurons) {
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons;
  fpSparseConVec = fopen("sparseConVec.dat", "rb");
  fpIdxVec = fopen("idxVec.dat", "rb");
  fpNpostNeurons = fopen("nPostNeurons.dat", "rb");
  //  int ;
  unsigned long int nConnections = 0, dummy = 0;
  printf("%p %p %p\n", fpIdxVec, fpNpostNeurons, fpSparseConVec);
  dummy = fread(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  for(unsigned int i = 0; i < N_NEURONS; ++i) {
    nConnections += nPostNeurons[i];
  }
  printf("#Post read\n #rec cons = %lu \n", nConnections);
  printf("sparseConVec address before: %p\n", sparseConVec);  
  sparseConVec = new int[nConnections] ;
  printf("sparseConVec address: %p\n", sparseConVec);
  dummy = fread(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  if(dummy != nConnections) {
    printf("sparseConvec read error ? \n");
  }
  printf("#sparse cons = %lu \n\n", dummy);    
  printf("sparse vector read\n");  
  dummy = fread(idxVec, sizeof(*idxVec), N_NEURONS, fpIdxVec);
  printf("#idx vector read\n");    
  fclose(fpSparseConVec);
  fclose(fpIdxVec);
  printf("sparseConVec address at LoadSparseConMat: %p\n", sparseConVec);      
}

void LoadRewiredCon(unsigned long nElements, unsigned int *IS_REWIRED_LINK) {
  unsigned long nElementsRead = 0;
  FILE *fpRewired;
  fpRewired = fopen("newPostNeurons.dat", "rb");
  nElementsRead = fread(IS_REWIRED_LINK, sizeof(*IS_REWIRED_LINK), nElements, fpRewired);
  printf("reading new post neurons\n");
  if(nElements != nElementsRead) {
    printf("rewired read error ? \n");
  }
  printf("done\n");
}

void GetFullMat(unsigned int *conVec, int* sparseVec, int* idxVec, int* nPostNeurons) {
 unsigned long long int i, j, k, nPost;
 // INITIALIZE
 for(i = 0; i < N_NEURONS; i++) {
   for(j = 0; j < N_NEURONS; j++) {
     conVec[i + N_NEURONS * j]  = 0;
   }
 }
 // CONSTRUCT
 for(i = 0; i < N_NEURONS; i++) {
   nPost = nPostNeurons[i];
   for(k = 0; k < nPost; k++) {
     j = sparseVec[idxVec[i] + k];
     conVec[i + N_NEURONS * j]  = 1; /* i --> j  */
    }
  }
}

void RemoveConnections(unsigned int *conVec, double kappa) {
 unsigned long long int i, j;
 double removalProb = kappa / sqrt((double)K);
 printf("removal prob = %f\n", removalProb);
 std::random_device rd;
 std::default_random_engine gen(rd());
 std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
 vector<unsigned long> nPost(NE);
 // COUNT BEFORE REMOVING CONNECTIONS
 for(i = 0; i < NE; i++) {
   nPost[i] = 0;
 } 
 for(j = 0; j < NE; j++) {
   for(i = 0; i < NE; i++) {
     if(conVec[i + N_NEURONS * j]) {
       nPost[j] += 1;
     }
   }
 }
 FILE *ffp;
 ffp = fopen("orig_K_EE_count.csv", "w");
 for(unsigned int lll = 0; lll < NE; lll++) {
   fprintf(ffp, "%lu\n", (unsigned long)nPost[lll]);
 }
 fclose(ffp); 
 // REMOVING CONNECTIONS
 for(i = 0; i < NE; i++) {
   for(j = 0; j < NE; j++) {
     if(conVec[i + N_NEURONS * j]) {
       if(removalProb >= UniformRand(gen)) {
	 conVec[i + N_NEURONS * j]  = 0;
       }
     }
   }
 }
 // COUNT AFTER REMOVING CONNECTIONS
 for(i = 0; i < NE; i++) {
   nPost[i] = 0;
 } 
 for(j = 0; j < NE; j++) {
   for(i = 0; i < NE; i++) {
     if(conVec[i + N_NEURONS * j]) {
       nPost[j] += 1;
     }
   }
 }
 ffp = fopen("removed_K_EE_count.csv", "w");
 for(unsigned int lll = 0; lll < NE; lll++) {
   fprintf(ffp, "%lu\n", (unsigned long)nPost[lll]);
 }
 fclose(ffp); 
}


void GenRewiredSparseMat(unsigned int *conVec,  unsigned int rows, unsigned int clms, int* sparseVec, int* idxVec, int* nPostNeurons, unsigned int *IF_REWIRED_CON, unsigned int *IS_REWIRED_LINK) {
  /* generate sparse representation
     conVec       : input vector / flattened matrix 
     sparseVec    : sparse vector
     idxVec       : every element is the starting index in sparseVec for ith row in matrix conVec
     nPostNeurons : number of non-zero elements in ith row 
  */
  printf("\n MAX Idx of conmat allowed = %u \n", rows * clms);
  unsigned long long int i, j, counter = 0, nPost;
  for(i = 0; i < rows; ++i) {
    nPost = 0;
    for(j = 0; j < clms; ++j) {
      // printf("%llu %llu %llu %llu\n", i, j, i + clms * j, i + rows * j);
      if(conVec[i + rows * j]) { /* i --> j  */
        sparseVec[counter] = j;
	if(IF_REWIRED_CON[i + rows * j] == 1) {
	  IS_REWIRED_LINK[counter] = 1;
	}
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


void AddConnections(unsigned int *conVec, double kappa, double rewiredEEWeight, int *sparseConVec, int *idxVec, int *nPostNeurons, unsigned int *IS_REWIRED_LINK) {
  unsigned long long int i, j, nConnections = 0, nElementsWritten;
  double addProb = 0; 
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<double> UniformRand(0.0, 1.0);
  vector<unsigned long> nPost(N_NEURONS);
  // READ PO's
  FILE *fpPOofNeurons = NULL, *ffp;
  fpPOofNeurons = fopen("poOfNeurons.dat", "rb");
  if(fpPOofNeurons == NULL) {
    printf("\n Error: file poOfNeurons.dat not found! \n");
    exit(1);
  }
  double *poOfNeurons = new double [NE];
  unsigned long int nPOsRead = 0; 
  nPOsRead = fread(poOfNeurons, sizeof(*poOfNeurons), NE, fpPOofNeurons);
  if(nPOsRead != NE) {
    printf("\n Error: All elements not written \n");
  }
  fclose(fpPOofNeurons);
  unsigned int *IF_REWIRED_CON = new unsigned int [(unsigned long int)N_NEURONS * N_NEURONS];
  for(i = 0; i < N_NEURONS; ++i) {
    for(j = 0; j < N_NEURONS; j++) {
      IF_REWIRED_CON[i + N_NEURONS * j] = 0;
    }
  }
  double denom = NE - K - kappa * sqrt(K);
  for(i = 0; i < NE; i++) {
    for(j = 0; j < NE; j++) {
      if(conVec[i + NE * j] == 0) {
	addProb = (kappa / rewiredEEWeight)* sqrt(K) * (1.0 + cos(2.0 * (poOfNeurons[i] - poOfNeurons[j]))) / denom;
	if(addProb > 1 || addProb < 0) { printf("IN AddConnections() add prob not in range!!!\n"); exit(1); }
	if(addProb >= UniformRand(gen)) {
	  conVec[i + N_NEURONS * j]  = 1;
	  IF_REWIRED_CON[i + N_NEURONS * j]  = 1;	    
	}
      }
    }
  }
  // COUNT AFTER ADDING CONNECTIONS
  for(i = 0; i < N_NEURONS; i++) {
    nPost[i] = 0;
  } 
  for(j = 0; j < N_NEURONS; j++) {
    for(i = 0; i < N_NEURONS; i++) {
      if(conVec[i + N_NEURONS * j]) {
	nPost[j] += 1;
	nConnections += 1;
      }
    }
  }
  ffp = fopen("rewired_K_count.csv", "w");
  for(unsigned int lll = 0; lll < NE; lll++) {
    fprintf(ffp, "%lu\n", (unsigned long)nPost[lll]);
  }
  fclose(ffp);
  // GENERATE SPARSE REPRESENTATION
  printf("#connections in rewired matrix = %llu\n", nConnections);
  cout << "computing sparse rep" << endl;
  sparseConVec = new int[nConnections];
  IS_REWIRED_LINK = new unsigned int[nConnections];
  GenRewiredSparseMat(conVec, N_NEURONS, N_NEURONS, sparseConVec, idxVec, nPostNeurons, IF_REWIRED_CON, IS_REWIRED_LINK);
  cout << "done" << endl;
  // WRITE TO FILE
  FILE *fpSparseConVec, *fpIdxVec, *fpNpostNeurons, *fpIsRewiredLink;
  printf("done\n #connections = %llu\n", nConnections);
  printf("writing to file ... "); fflush(stdout);
  // remove old dat symlink 
  if(remove("sparseConVec.dat") != 0) { perror( "Error deleting sparseconvec.dat" ); }
  else { puts( "File successfully deleted" ); }
  fpSparseConVec = fopen("sparseConVec.dat", "wb");
  nElementsWritten = fwrite(sparseConVec, sizeof(*sparseConVec), nConnections, fpSparseConVec);
  fclose(fpSparseConVec);
  if(nElementsWritten != nConnections) {
    printf("\n Error: All elements not written \n");
  }
 
  printf("writing new cons to file ... "); fflush(stdout);
  fpIsRewiredLink = fopen("newPostNeurons.dat", "wb");
  nElementsWritten = 0;
  nElementsWritten = fwrite(IS_REWIRED_LINK, sizeof(*IS_REWIRED_LINK), nConnections, fpIsRewiredLink);
  fclose(fpIsRewiredLink);
  if(nElementsWritten != nConnections) {
    printf("\n Error: All elements not written \n");
  }
 
  if(remove("idxVec.dat") != 0) { perror( "Error deleting sparseconvec.dat" ); }
  else { puts( "File successfully deleted" ); }
  fpIdxVec = fopen("idxVec.dat", "wb");
  fwrite(idxVec, sizeof(*idxVec), N_NEURONS,  fpIdxVec);
  fclose(fpIdxVec);
  if(remove("nPostNeurons.dat") != 0) { perror( "Error deleting sparseconvec.dat" ); }
  else { puts( "File successfully deleted" ); }
  fpNpostNeurons = fopen("nPostNeurons.dat", "wb");
  fwrite(nPostNeurons, sizeof(*nPostNeurons), N_NEURONS, fpNpostNeurons);
  fclose(fpNpostNeurons);
  printf("done\n");
  delete [] poOfNeurons;
  for(i = 0; i < NE; i++) {
    nPost[i] = 0;
  } 
  for(j = 0; j < NE; j++) {
    for(i = 0; i < NE; i++) {
      if(conVec[i + N_NEURONS * j]) {
	nPost[j] += 1;
      }
    }
  }
  FILE *ffp22;
  ffp22 = fopen("rewired_K_EE_count.csv", "w");
  for(unsigned int lll = 0; lll < NE; lll++) {
    fprintf(ffp22, "%lu\n", (unsigned long)nPost[lll]);
  }
  fclose(ffp22);
  delete [] IF_REWIRED_CON;
}


int main(int argc, char *argv[]) {
  double phi_ext, kappa, rewiredEEWeight;
  int *nPostNeurons =NULL, *idxVec = NULL;  
  if(argc > 1) {
      phi_ext = atof(argv[1]);
  }
  if(argc > 2) {
      kappa = atof(argv[2]);
  }
  if(argc > 3) {
      rewiredEEWeight = atof(argv[3]);
  }
  
  if(IF_REWIRE) {

    unsigned int *IS_REWIRED_LINK = NULL;
    // int *
    
    nPostNeurons = new int[N_NEURONS];
    idxVec = new int[N_NEURONS];

    printf("\n----------------------------------------\n");
    printf("---------------REWIRING-----------------\n");          
    printf("----------------------------------------\n");      
    printf("loading Sparse matrix\n");  
    LoadSparseConMat(idxVec, nPostNeurons);
    printf("sparseConVec address after LoadSparseConMat: %p\n", sparseConVec);    
    unsigned long nEE_IEConnections = 0;
    for(unsigned i = 0; i < N_NEURONS; i++) {
      nEE_IEConnections += nPostNeurons[i];
    }
    IS_REWIRED_LINK = new unsigned int[nEE_IEConnections];    
    if(IF_REWIRE && phi_ext > 0) {
      LoadRewiredCon(nEE_IEConnections, IS_REWIRED_LINK);
    }
    else {
      for(unsigned long i = 0; i < nEE_IEConnections; i++) {
    	IS_REWIRED_LINK[i] = 0;
      }
    }    
    if(phi_ext == 0) { // REWIRE ONLY ONCE FOR ALL ANGLES
      unsigned int *conMatCONSTRUCTED = new unsigned int [(unsigned long int)N_NEURONS * N_NEURONS];
      printf("\n reconstructing matrix... sparseVec addrs%p", sparseConVec);  fflush(stdout);     
      GetFullMat(conMatCONSTRUCTED, sparseConVec, idxVec, nPostNeurons);
      printf("done\n");
      printf("\n removing connections... ");  fflush(stdout);  
      RemoveConnections(conMatCONSTRUCTED, kappa);
      printf("done\n");
      if(sparseConVec != NULL) {
      	printf("\ndeleting old sparseConVec\n");
      	delete [] sparseConVec;
      	sparseConVec = NULL;
      }
      if(IS_REWIRED_LINK != NULL) {
      	printf("\ndeleting old sparseConVec\n");
      	delete [] IS_REWIRED_LINK;
      	IS_REWIRED_LINK = NULL;
      }      
      printf("\n adding connections... ");  fflush(stdout);    
      AddConnections(conMatCONSTRUCTED, kappa, rewiredEEWeight, sparseConVec, idxVec, nPostNeurons, IS_REWIRED_LINK);  
      printf("done\n");
      // delete [] conMat;
      delete [] conMatCONSTRUCTED;
    }
      if(sparseConVec != NULL) {
      	printf("\ndeleting sparseConVec\n");
      	delete [] sparseConVec;
      	sparseConVec = NULL;
      }
  delete [] IS_REWIRED_LINK;
  delete [] nPostNeurons;
  delete [] idxVec;
  delete [] sparseConVec;
  }

  return EXIT_SUCCESS;
}
