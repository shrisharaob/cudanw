#ifndef _READCONMATFROMFILE_
#define _READCONMATFROMFILE_
#include <stdio.h>
/* read connection matrix into global variable conMat from text file */
void ReadConMatFromFile(const char* filename, int nNeurons) {
  FILE* fp;
  int i, j;
  int buffer[nNeurons * nNeurons];
  /*  size_t result;*/
  
  /*  long fileSize;*/
  fp = fopen(filename, "r");

  /*  fseek(fp, 0, SEEK_END);
  fileSize = ftell(fp);
  rewind(fp);
  buffer = (char *)malloc(sizeof(char) * fileSize);
  if(buffer == NULL) {
  fputs("buffer not allocated", stderr);
  exit(1);
  }
  */
  if(fp != NULL) {
    result = fread(buffer, sizeof(int), nNeurons * nNeurons, fp);
    for(i = 0; i < nNeurons; ++i) {
      for(j = 0; j < nNeurons; ++j) {
        conVec[i * nNeurons + j] = (double)(buffer[i + j * nNeurons] - '0'); /*convert ascii decimal character representation to integer*/
      }
    }
    fclose(fp);
  }
  else {
    printf("\n%s does not exist\n", filename);
  }
}
#endif





