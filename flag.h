#ifndef _FLAG_H
#define _FLAG_H
#include "MTTKRP.h"
#include <climits>
#include <cstring>
#include <limits>
#include <stdint.h>
using std::numeric_limits;


template<typename T, typename type_thread>
class flag {
 public:
  flag(semiTensor<T> tensor, int BLOCK_SIZE);

  type_thread *cflag;
  type_thread *bit_flag;
  int *first;
  uint8_t *startflag;
  uint8_t *block_flag;
};



template<typename T, typename type_thread>
flag<T, type_thread>::flag(semiTensor<T> tensor, int BLOCK_SIZE) {
  int threadLen = sizeof(type_thread) * 8;
  int nnz = tensor.nnz;

  int flagLen = (nnz - 1) / threadLen + 1;


  int Gridsize = (flagLen - 1) / BLOCK_SIZE + 1;

  cflag = (type_thread *)malloc(sizeof(type_thread) * flagLen);
  bit_flag = (type_thread *)malloc(sizeof(type_thread) * flagLen);
  first = (int *)malloc(sizeof(int) * flagLen);
  startflag = (uint8_t *)malloc(sizeof(uint8_t) * flagLen);
  block_flag = (uint8_t *)malloc(sizeof(uint8_t) * Gridsize);

  memset(bit_flag, -1, sizeof(type_thread)*flagLen);
  memset(cflag, 0, sizeof(type_thread)*flagLen);
  memset(startflag, 0, sizeof(uint8_t)*flagLen);
  memset(block_flag, 0, sizeof(uint8_t)*Gridsize);
  for (int i = 0; i < flagLen; i++) {
    // T ibits;
    T bits = 0;
    for (int j = 0; j < threadLen && (i * threadLen + j) < nnz; j++) {
      type_thread elem = tensor.flag[i * threadLen + j];
      bits += (elem << j);
    }
    cflag[i] = bits;

  }


  for (int i = 0; i < flagLen - 1; i++) {
    type_thread ibits = 0;
    for (int j = 0; j < threadLen; j++) {
      unsigned int nextelem = tensor.flag[i * threadLen + j + 1];

      if (nextelem == 1) {
        ibits += ((type_thread)1 << j);
      }
    }
    bit_flag[i] -= ibits;

  }

  type_thread ibits = 0;
  for (int j = 0; j < threadLen; j++) {
    int index = (flagLen - 1) * threadLen + j + 1;
    unsigned int nextelem;
    if (index < nnz) {
      nextelem = tensor.flag[index];

      if (nextelem == 1) {
        ibits += ((type_thread)1 << j);
      }
    }

    if (j == threadLen - 1) {
      ibits += ((type_thread)1 << j);
    }

  }


  bit_flag[flagLen - 1] -= ibits;
  


  for (int i = 0; i < flagLen; i++) {
    if (bit_flag[i] != numeric_limits<type_thread>::max()) {
      startflag[i] = 1;
    }

  }

  first[0] = -1;
  for (int i = 0; i < flagLen - 1; i++) {
    int elem = tensor.flag[i * threadLen];
    if (elem == 1) {
      ++first[i];
    }
    int sum = first[i];
    for (int j = 1; j < threadLen; j++) {
      elem = tensor.flag[i * threadLen + j];
      if (elem == 1) {
        ++sum;
      }
    }
    first[i + 1] = sum;
  }
  if (tensor.flag[(flagLen - 1)*threadLen] == 1) {
    ++first[flagLen - 1];
  }


  for (int i = 0; i < Gridsize - 1; i++) {
    uint8_t *val = startflag + i * BLOCK_SIZE;
    for (int j = 0; j < BLOCK_SIZE; j++) {
      if (val[j] == 1) {
        block_flag[i] = 1;
        break;
      }
    }
  }

  int baseindex = (Gridsize - 1) * BLOCK_SIZE;
  uint8_t *val = startflag + baseindex;
  for (int j = 0; j < BLOCK_SIZE && baseindex + j < flagLen; j++) {
    if (val[j] == 1) {
      block_flag[Gridsize - 1] = 1;
      break;
    }
  }
}

#endif