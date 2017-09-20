#ifndef _TTMGPU_H
#define _TTMGPU_H
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include <device_functions.h>
#include <stdio.h>
#include <stdint.h>
#include "flag.h"
#include "MTTKRP.h"
#include "convert.h"


template <typename T>
struct tensor_gpu {

  unsigned int *d_j;
  unsigned int *d_k;
  T *d_val;
  int d_nnz;  //length
  int d_threadlen; // the length of data single cuda thread
  int d_iterlen;  // length/d_threadlen
  tensor_gpu(stensor H_tensor, int thread_len);
  void Free(int a = 0);
  ~tensor_gpu();
};

template <typename T, typename type_thread>
struct semitensor_gpu {
 public:
  int d_nfibs;
  int d_nCols;
  int d_nnz;
  type_thread *d_bflags;
  int *d_first; // each thread first result; index for recahce
  T *d_val;
  T *d_last_partial;
  T *d_recache;
  T *d_blockSum;
  T *h_result;

  int BLOCK_SIZE;


  uint8_t *d_startflag;
  // unsigned short *d_startflag_backup;
  uint8_t *d_blockflag;
  semitensor_gpu(semiTensor<T> H_tensor, flag<T, type_thread> h_flag, int BLOCK_SIZE);
  void Free(int a = 0);
  ~semitensor_gpu();
};



template <typename T>
tensor_gpu<T>:: tensor_gpu(stensor H_tensor, int thread_len) {
  d_nnz = ((H_tensor.nnz - 1) / thread_len + 1) * thread_len;
  d_threadlen = thread_len;
  d_iterlen = d_nnz / d_threadlen;

  cudaMalloc((void **)&d_j, sizeof(int)*d_nnz);
  cudaMalloc((void **)&d_k, sizeof(int)*d_nnz);
  cudaMalloc((void **)&d_val, sizeof(T)*d_nnz);

  cudaMemset(d_j, 0, sizeof(int)*d_nnz);
  cudaMemset(d_k, 0, sizeof(int)*d_nnz);
  cudaMemset(d_val, 0, sizeof(T)*d_nnz);


  cudaMemcpy(d_j, H_tensor.j, sizeof(int)*d_nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, H_tensor.k, sizeof(int)*d_nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, H_tensor.val, sizeof(T)*d_nnz, cudaMemcpyHostToDevice);
}

template <typename T>
void tensor_gpu<T>::Free(int a) {
  cudaFree(d_j);
  cudaFree(d_k);
  cudaFree(d_val);
}

template <typename T>
tensor_gpu<T>::~tensor_gpu() {

}

template <typename T, typename type_thread>
semitensor_gpu<T, type_thread>::semitensor_gpu(semiTensor<T> H_tensor,
                                               flag<T, type_thread> h_flag,
                                               int BLOCK_SIZE) {
  d_nfibs = H_tensor.nfibs;

  d_nCols = H_tensor.R;

  this->BLOCK_SIZE = BLOCK_SIZE;

 
  int bit_len = sizeof(type_thread) * 8;
  d_nnz = ((H_tensor.nnz - 1) / bit_len + 1);

  printf("##TTEST %d\n", d_nnz);
  int nBlock = ((d_nnz - 1) / BLOCK_SIZE + 1);

  cudaMalloc((void **)&d_val, sizeof(T)*d_nfibs * d_nCols);
  cudaMemset(d_val, 0, sizeof(T)*d_nfibs * d_nCols);

  cudaMalloc((void **)&d_bflags, sizeof(type_thread)*d_nnz);
  cudaMemcpy(d_bflags, h_flag.bit_flag, sizeof(type_thread)*d_nnz, cudaMemcpyHostToDevice);


  cudaMalloc((void **)&d_first, sizeof(int)*d_nnz);
  cudaMemcpy(d_first, h_flag.first, sizeof(int)*d_nnz, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_last_partial, sizeof(T)*d_nnz * d_nCols);
  cudaMemset(d_last_partial, 0, sizeof(T)*d_nnz * d_nCols);

  cudaMalloc((void **)&d_recache, sizeof(T)*d_nfibs * d_nCols);
  cudaMemset(d_recache, 0, sizeof(T)*d_nfibs * d_nCols);


  h_result = (T *)malloc(sizeof(T) * d_nfibs * d_nCols);
  memset(h_result, 0, sizeof(T)*d_nfibs * d_nCols);


  cudaMalloc((void **)&d_startflag, sizeof(uint8_t)*d_nnz);

    cudaMemcpy(d_startflag , h_flag.startflag, sizeof(uint8_t)*d_nnz, cudaMemcpyHostToDevice);


  cudaMalloc((void **)&d_blockSum, sizeof(T)*nBlock * d_nCols);
  cudaMemset(d_blockSum, -1, sizeof(T)*nBlock * d_nCols);

  cudaMalloc((void **)&d_blockflag, sizeof(uint8_t)*nBlock);
  cudaMemcpy(d_blockflag, h_flag.block_flag, sizeof(uint8_t)*nBlock, cudaMemcpyHostToDevice);

}

template <typename T, typename type_thread>
void semitensor_gpu<T, type_thread>::Free(int a) {
  cudaFree(d_val);
  cudaFree(d_bflags);
  cudaFree(d_first);
  cudaFree(d_last_partial);
  cudaFree(d_recache);
  cudaFree(d_startflag);
  cudaFree(d_blockSum);
  cudaFree(d_blockflag);

}
template <typename T, typename type_thread>
semitensor_gpu<T, type_thread>::~semitensor_gpu() {

}

template <typename T>
void transpose_tensor(soa_tensor<T> &H_tensor, int threadLen) {
  int nnz = H_tensor.nnz;

  int flagLen = (nnz - 1) / threadLen + 1;

  unsigned int *H_j = (unsigned int *)malloc(sizeof(unsigned int) * flagLen * threadLen);
  unsigned int *H_k = (unsigned int *)malloc(sizeof(unsigned int) * flagLen * threadLen);
  T *H_val = (T *)malloc(sizeof(T) * flagLen * threadLen);

  // memset(H_i,0,sizeof(int)*flagLen*threadLen);
  memset(H_j, 0, sizeof(int)*flagLen * threadLen);
  memset(H_k, 0, sizeof(int)*flagLen * threadLen);
  memset(H_val, 0, sizeof(T)*flagLen * threadLen);

  for (int i = 0; i < nnz; i++) {
    int row = i / threadLen;
    int col = i % threadLen;

    int index = col * flagLen + row;
    // H_i[index]=H_tensor.i[i];
    H_j[index] = H_tensor.j[i];
    H_k[index] = H_tensor.k[i];
   
    H_val[index] = H_tensor.val[i];
  }

  // free(H_tensor.i);
  free(H_tensor.j);
  free(H_tensor.k);
  free(H_tensor.val);

  // H_tensor.i=H_i;
  H_tensor.j = H_j;
  H_tensor.k = H_k;
  H_tensor.val = H_val;
}

template <typename T>
void transpose_matrix(T *&val, int nRows, int nCols) {
  T *tmp = (T *)malloc(sizeof(T) * nRows * nCols);
  for (int i = 0; i < nRows; i++)
    for (int j = 0; j < nCols; j++) {
      tmp[j * nRows + i] = val[i * nCols + j];
    }

  free(val);
  val = tmp;
}


mtype *callTTM(stensor ltensor, mtype *B_matrix, mtype *C_matrix, int B_nRows, int C_nRows, int nCols, semitensor rtensor, type_thread threadtype, int blocksize);

#endif