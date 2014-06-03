#include "globalVars.h"
#include "devFunctionProtos.h"

__device__ void rk4(float *y, float *dydx, int n, float rk4X, float h, float *yout, float iSynap)
{
	int i;
	float xh, hh, h6, dym[N_STATEVARS], dyt[N_STATEVARS], yt[N_STATEVARS];
	hh = h*0.5;
	h6 = h/6.0;
	xh = rk4X+hh;
	for (i = 0; i < n; i++) { /* 1st step */
      yt[i] = y[i] + hh * dydx[i]; 
    }
	derivs(xh,yt,dyt, iSynap);                     /* 2nd step */
	for (i = 0; i < n; i++) yt[i] = y[i] + hh * dyt[i];
	derivs(xh,yt,dym, iSynap);                     /* 3rd step */
	for (i = 0; i < n; i++)
	{
		yt[i]=y[i]+h*dym[i];
		dym[i] += dyt[i];
	}
	derivs(rk4X+h,yt,dyt, iSynap);                    /* 4th step */
	for (i = 0; i < n; i++) yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}
