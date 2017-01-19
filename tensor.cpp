#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <stdio.h>
#include "splatt.h"
#include "readtensor.h"
#include "convert.h"
#include "matrix.h"
#include "test.h"
#include "MTTKRP.h"
#include "gpuMTTKRP.h"

using std::cout;
using std::cerr;
using std::endl;

using std::string;
// using std::vector;

using std::sort;




class sorter {
 public:
  sorter (const int &_sort_order): sort_order(_sort_order) {}

  bool operator()(const item &one, const item &two) const {

    int idx0 = (sort_order) % 3;
    int idx1 = (idx0 + 1) % 3;
    int idx2 = (idx0 + 2) % 3;

    if (one.coord[idx0] != two.coord[idx0]) {
      return one.coord[idx0]  < two.coord[idx0];
    } else if (one.coord[idx1] != two.coord[idx1]) {
      return one.coord[idx1] < two.coord[idx1];
    } else {
      return one.coord[idx2] < two.coord[idx2];
    }
    return false;
  }
 private:
  int sort_order;
};


int main(int argc, char **argv) {
  int dim_i = 0;
  int dim_j = 0;
  int dim_k = 0;

  int R = 16;
  int BLOCK_SIZE = 1024;
  int mode=0;

  char *in_file = argv[1];
  if (argc > 2 and atoi(argv[2])) {
    R = atoi(argv[2]);
  }
  if (argc > 3 and atoi(argv[3])) {
    BLOCK_SIZE = atoi(argv[3]);
  }

   if(argc > 4 and atoi(argv[4])){
    mode=atoi(argv[4]);

    if(mode>=3){
      printf("Error, the mode parameter is only allowed into 0-2\n");
    }
  }

  struct timeval start;
  struct timeval end;
  int nRows1, nRows2;

  int nnz = precess(dim_i, dim_j, dim_k, argv[1]);
  printf("I=%d J=%d k=%d\n", dim_i, dim_j, dim_k);  // TTM multiply on third dimension, MTTKRP multiply on second and third dimension

  tensor data;
  tensor_malloc(&data, nnz);

  readtensor(data, in_file);

  sorter compare(0);
  sort(data,data+nnz,compare);
  // test(data, nnz);

  stensor H_Tensor(nnz);
  convert(data, H_Tensor, nnz,mode);

  // ttype *TTM_matrix;
  // genMatrix(&TTM_matrix, dim_k + 1, R);
  // randomFill(TTM_matrix, dim_k + 1, R);

  ttype *A,*B, *C; // B,C for MTTKRP

  if(mode==0){
    nRows1=dim_j+1;
    nRows2=dim_k+1;
  }
  else if(mode==1){
    nRows1=dim_k+1;
    nRows2=dim_i+1;
  }
  else {
    nRows1=dim_i+1;
    nRows2=dim_j+1;
  }

 

  // genMatrix(&A, dim_i + 1, R);
  // randomFill(A, dim_i + 1, R);
  genMatrix(&B, nRows1, R);
  randomFill(B, nRows1, R);
  genMatrix(&C, nRows2, R);
  randomFill(C, nRows2, R);
  // test1(H_Tensor);
  // matrices[0]=A;
  // matrices[1]=B;
  // matrices[2]=C;


  semitensor rtensor;
  int *flag;
  int nfibs = preprocess(H_Tensor, &flag, rtensor);


  printf("nfibs=%d\n", nfibs);
  gettimeofday(&start, NULL);
  MTTKRP(H_Tensor, nfibs, B, C, R, rtensor);
  gettimeofday(&end, NULL);

  float CPU_time2 = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
  // test_TTM(rtensor);
  unsigned char type = 0;

  printf("CPU_time MTTKRP: %f\n", CPU_time2);
  ttype *d_result = callTTM(H_Tensor, B, C, nRows1, nRows2, R, rtensor, type, BLOCK_SIZE);


  printf("%s %d\n", __FILE__, __LINE__);
  verify(rtensor, d_result);
  printf("%s %d\n", __FILE__, __LINE__);
  tensor_free(data);
  printf("%s %d\n", __FILE__, __LINE__);
  // delete [] B;
  // delete [] C;
  printf("%s %d\n", __FILE__, __LINE__);
  // delete [] TTM_matrix;
  printf("%s %d\n", __FILE__, __LINE__);
  // free(B);
  // free(C);
  // free(TTM_matrix);
  // freeMatrix(&TTM_matrix);
  // freeMatrix(&B);
  // freeMatrix(&C);
}

