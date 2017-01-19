#include "MTTKRP.h"

int preprocess(stensor htensor, int **flag, semitensor &result) {
  int nnz = htensor.nnz;
  *flag = (int *)malloc(sizeof(int) * nnz);
  memset(*flag, 0, sizeof(int)*nnz);
  int prev_i = htensor.i[0];
  int prev_j = htensor.j[0];
  // int prev_j=htensor.j[0];
  (*flag)[0] = 1;
  int nfibs = 1;
  for (int i = 1; i < nnz; i++) {
    // if(i<8)printf("i=%d###Bangtian %d %d prev_i=%d prev_j=%d###\n", i,htensor.i[i],htensor.j[i],prev_i,prev_j);
    if (htensor.i[i] != prev_i) {
      (*flag)[i] = 1;
      ++nfibs;
      prev_i = htensor.i[i];
      prev_j = htensor.j[i];
    }

  }
  result.nfibs = nfibs;
  result.flag = *flag;
  result.nnz = nnz;
  result.i = (int *)malloc(sizeof(int) * nfibs);
  // result.j=(int *)malloc(sizeof(int)*nfibs);

  return nfibs;

}


void MTTKRP(stensor htensor,
            int nfibs,
            ttype *B,
            ttype *C,
            int nCols,
            semitensor &result) {
  int nnz = htensor.nnz;
  result.R = nCols;
  ttype *tmp = (ttype *)malloc(sizeof(ttype) * nfibs * nCols);
  memset(tmp, 0, sizeof(ttype)*nfibs * nCols);
  int index = -1;
  for (int i = 0; i < nnz; ++i) {
    unsigned int j = htensor.j[i];
    unsigned int k = htensor.k[i];
    ttype val = htensor.val[i];

    if (result.flag[i] == 1) {
      ++index;
      result.i[index] = htensor.i[i];
      // result.j[index]=htensor.j[i];
    }

    for (int r = 0; r < nCols; r++) {
      tmp[index * nCols + r] += val * B[j * nCols + r] * C[k * nCols + r];
    }     /* code */
  }
  result.val = tmp;
}