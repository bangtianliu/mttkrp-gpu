#ifndef _TTM_H
#define _TTM_H
#include "convert.h"
#include <string.h>

#ifdef DOUBLE
typedef double ttype;
#else
typedef float ttype;
#endif

template <typename T>
struct semiTensor {
 public:
  int *i;
  // int *j;
  int nfibs;
  int nnz;
  int *flag;
  int R;
  T *val;
};

#ifdef DOUBLE
typedef semiTensor<double> semitensor;

// typedef soa_tensor<double> *tensor;
#else
typedef semiTensor<float> semitensor;
#endif

int preprocess(stensor htensor, int **flag, semitensor &result);


void MTTKRP(stensor htensor,
            int nfibs,
            ttype *B,
            ttype *C,
            int nCols,
            semitensor &result);

#endif