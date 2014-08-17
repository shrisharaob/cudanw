#ifndef _RK4_
#define _RK4_
#include "globalVars.h"
#include "devFunctionProtos.h"

__device__ void rk4(double *y, double *dydx, int n, double rk4X, double h, double *yout, double iSynap, double ibg, double iff)
{
	unsigned int i;
	double xh, hh, h6, dym[N_STATEVARS], dyt[N_STATEVARS], yt[N_STATEVARS];
	hh = h*0.5;
	h6 = h/6.0;
	xh = rk4X+hh;
	for (i = 0; i < n; i++) { /* 1st step */
      yt[i] = y[i] + hh * dydx[i]; 
    }
	derivs(xh,yt,dyt, iSynap, ibg, iff);                     /* 2nd step */
	for (i = 0; i < n; i++) yt[i] = y[i] + hh * dyt[i];
	derivs(xh,yt,dym, iSynap, ibg, iff);                     /* 3rd step */
	for (i = 0; i < n; i++)
	{
		yt[i]=y[i]+h*dym[i];
		dym[i] += dyt[i];
	}
	derivs(rk4X+h,yt,dyt, iSynap, ibg, iff);                    /* 4th step */
	for (i = 0; i < n; i++) yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}
#endif
