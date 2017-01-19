#include "convert.h"

void convert(tensor data, stensor &CPU_tensor, int nnz, int mode) {
  int index0=mode;
 int index1=(mode+1)%3;
 int index2=(mode+2)%3;
  for (int i = 0; i < nnz; i++) {
    // printf("Intest %d\n", data[i].coord[0]);
    CPU_tensor.i[i] = data[i].coord[index0];
    CPU_tensor.j[i] = data[i].coord[index1];
    CPU_tensor.k[i] = data[i].coord[index2];
    CPU_tensor.val[i] = data[i].val;
    CPU_tensor.nnz = nnz;
  }
}