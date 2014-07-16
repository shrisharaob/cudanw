#ifndef _AUX_
#define _AUX_
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "globalVars.h"
#include "devFunctionProtos.h"
#include "cudaRandFuncs.cu"
/* #include "rkdumb.cu" // NVIDIA provides no linker so have to include SOURCE FILES to keep files of managble size */
#include "isynap.cu"
#include "rkdumbPretty.cu"
#include "GenSparseMat.cu"
#include "bgCurrent.cu"
#include "IFF.cu"
#endif
