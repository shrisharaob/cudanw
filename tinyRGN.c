#ifndef __TINY_RNG__
#define __TINY_RNG__
#include <stdint.h>
#include <time.h>
#include <stdio.h>

#include <inttypes.h>


#define TINYMT32_MEXP 127
#define TINYMT32_SH0 1
#define TINYMT32_SH1 10
#define TINYMT32_SH8 8
#define TINYMT32_MASK UINT32_C(0x7fffffff)
#define TINYMT32_MUL (1.0f / 4294967296.0f)
#define MIN_LOOP 8
#define PRE_LOOP 8


#ifndef __CONCAT
#define __CONCATenate(left, right) left ## right
#define __CONCAT(left, right) __CONCATenate(left, right)
#endif

#define UINT32_C(value)   __CONCAT(value, UL)


struct TINYMT32_T {
    uint32_t status[4];
    uint32_t mat1;
    uint32_t mat2;
    uint32_t tmat;
};

typedef struct TINYMT32_T tinymt32_t;

inline static void tinymt32_next_state(tinymt32_t * random) {
    uint32_t x;
    uint32_t y;

    y = random->status[3];
    x = (random->status[0] & TINYMT32_MASK)
	^ random->status[1]
	^ random->status[2];
    x ^= (x << TINYMT32_SH0);
    y ^= (y >> TINYMT32_SH0) ^ x;
    random->status[0] = random->status[1];
    random->status[1] = random->status[2];
    random->status[2] = x ^ (y << TINYMT32_SH1);
    random->status[3] = y;
    random->status[1] ^= -((int32_t)(y & 1)) & random->mat1;
    random->status[2] ^= -((int32_t)(y & 1)) & random->mat2;
}

static uint32_t ini_func1(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1664525);
}


static uint32_t ini_func2(uint32_t x) {
    return (x ^ (x >> 27)) * UINT32_C(1566083941);
}

static void period_certification(tinymt32_t * random) {
    if ((random->status[0] & TINYMT32_MASK) == 0 &&
	random->status[1] == 0 &&
	random->status[2] == 0 &&
	random->status[3] == 0) {
	random->status[0] = 'T';
	random->status[1] = 'I';
	random->status[2] = 'N';
	random->status[3] = 'Y';
    }
}




void tinymt32_init(tinymt32_t * random, uint32_t seed);
void tinymt32_init(tinymt32_t * random, uint32_t seed) {
  int i;
  random->status[0] = seed;
    random->status[1] = random->mat1;
    random->status[2] = random->mat2;
    random->status[3] = random->tmat;
    for (i = 1; i < MIN_LOOP; i++) {
	random->status[i & 3] ^= i + UINT32_C(1812433253)
	    * (random->status[(i - 1) & 3]
	       ^ (random->status[(i - 1) & 3] >> 30));
    }
    period_certification(random);
    for (i = 0; i < PRE_LOOP; i++) {
	tinymt32_next_state(random);
    }
}







inline static uint32_t tinymt32_temper(tinymt32_t * random) {
    uint32_t t0, t1;
    t0 = random->status[3];
#if defined(LINEARITY_CHECK)
    t1 = random->status[0]
	^ (random->status[2] >> TINYMT32_SH8);
#else
    t1 = random->status[0]
	+ (random->status[2] >> TINYMT32_SH8);
#endif
    t0 ^= t1;
    t0 ^= -((int32_t)(t1 & 1)) & random->tmat;
    return t0;
}



inline static float tinymt32_generate_float(tinymt32_t * random) {
    tinymt32_next_state(random);
    return tinymt32_temper(random) * TINYMT32_MUL;
}






/* int main(int argc, const char* argv[]) */
/* { */
/*   tinymt32_t state; */
/*   uint32_t seed = time(0); */
/*   int i; */
/*   tinymt32_init(&state, seed); */
/*   FILE *fp = fopen("tst.txt", "w"); */
/*   for (i=0; i<10000; i++) */
/*     fprintf(fp, "%f\n", tinymt32_generate_float(&state)); */
/*   //  printf("random number %d: %f\n", i, tinymt32_generate_float(&state)); */
  
/* } */

#endif
