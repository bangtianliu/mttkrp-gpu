#ifndef _FLAG_H
#define _FLAG_H
#include "MTTKRP.h"
#include <climits>
#include <cstring>
#include <limits>
using std::numeric_limits;


template<typename T, typename type_thread>
class flag {
 public:
  flag(semiTensor<T> tensor, int BLOCK_SIZE);
  // type_thread *getFlag();
  // T *getBFlag();
  type_thread *cflag;
  type_thread *bit_flag;
  int *first;
  unsigned short *startflag;
  unsigned short *block_flag;
};


// template<typename T>
// inline T *flag<T>::getFlag(){
//  return this->flag;
// }

// template<typename T>
// inline T *flag<T>::getBFlag(){
//  return this->bit_flag;
// }

template<typename T, typename type_thread>
flag<T, type_thread>::flag(semiTensor<T> tensor, int BLOCK_SIZE) {
  int threadLen = sizeof(type_thread) * 8;
  int nnz = tensor.nnz;

  // printf("Test nnz: %s %s %d %d\n", __FILE__, __FUNCTION__, __LINE__, nnz);

  int flagLen = (nnz - 1) / threadLen + 1;

  // flagLen = 32-flagLen%32+flagLen;
  // int BLOCK_SIZE=tensor.BLOCK_SIZE;
  int Gridsize = (flagLen - 1) / BLOCK_SIZE + 1;

  cflag = (type_thread *)malloc(sizeof(type_thread) * flagLen);
  bit_flag = (type_thread *)malloc(sizeof(type_thread) * flagLen);
  first = (int *)malloc(sizeof(int) * flagLen);
  startflag = (unsigned short *)malloc(sizeof(unsigned short) * flagLen);
  block_flag = (unsigned short *)malloc(sizeof(unsigned short) * Gridsize);

  memset(bit_flag, -1, sizeof(type_thread)*flagLen);
  memset(cflag, 0, sizeof(type_thread)*flagLen);
  memset(startflag, 0, sizeof(unsigned short)*flagLen);
  memset(block_flag, 0, sizeof(unsigned short)*Gridsize);
  for (int i = 0; i < flagLen; i++) {
    // T ibits;
    T bits = 0;
    for (int j = 0; j < threadLen && (i * threadLen + j) < nnz; j++) {
      type_thread elem = tensor.flag[i * threadLen + j];
      bits += (elem << j);
    }
    cflag[i] = bits;

  }

  // printf("Flag: %d %d\n", cflag[0],flagLen);
  for (int i = 0; i < flagLen - 1; i++) {
    type_thread ibits = 0;
    for (int j = 0; j < threadLen; j++) {
      unsigned int nextelem = tensor.flag[i * threadLen + j + 1];
      // printf("nextelem: %s %d %d\n", __FILE__,__LINE__, nextelem);
      if (nextelem == 1) {
        ibits += ((type_thread)1 << j);
      }
    }
    bit_flag[i] -= ibits;

    // printf("Bangtian Liu:i=%d %x\n", i,bit_flag[i]);
  }

  type_thread ibits = 0;
  for (int j = 0; j < threadLen; j++) {
    int index = (flagLen - 1) * threadLen + j + 1;
    unsigned int nextelem;
    if (index < nnz) {
      nextelem = tensor.flag[index];
      // printf("nextelem: j=%d %s %d %d %d\n", j,__FILE__,__LINE__, nextelem,nnz);
      if (nextelem == 1) {
        ibits += ((type_thread)1 << j);
      }
    }
    // if(index==nnz-1){
    //  ;
    //  // ibits+=(1<<j);
    // }
    // if(index>nnz-1&&j<threadLen-1){
    //  ;
    // }
    if (j == threadLen - 1) {
      ibits += ((type_thread)1 << j);
    }

  }

  // printf("Test ibits: %s %d %x\n", __FILE__,__LINE__, ibits);
  bit_flag[flagLen - 1] -= ibits;
  // printf("Test bits: %s %d %x\n", __FILE__,__LINE__, bit_flag[flagLen-1]);


  for (int i = 0; i < flagLen; i++) {
    if (bit_flag[i] != numeric_limits<type_thread>::max()) {
      startflag[i] = 1;
    }
    // printf("####test startflag %d\n", startflag[i]);
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
  // first[0]=0; // first result entry on each thread
  // for(int i=1;i<flagLen;i++)
  // {
  //  int sum=first[i-1];
  //  for(int j=0;j<threadLen-1;j++)
  //  {
  //    int elem=tensor.flag[(i-1)*threadLen+j];
  //    int nextelem=tensor.flag[(i-1)*threadLen+j+1];
  //    if(elem==0&&nextelem==1)++sum;
  //    // if(tensor.flag[(i-1)*threadLen+j]==1){
  //    //  ++sum;
  //    // }
  //  }
  //  // printf("%s %d before i=%d first=%d\n", __FUNCTION__,__LINE__,i,sum);
  //  int elem=tensor.flag[(i-1)*threadLen+threadLen-1];
  //  int nextelem=tensor.flag[i*threadLen];
  //  if(elem==0&&nextelem==1)++sum;
  //     printf("%s %d i=%d first=%d\n", __FUNCTION__,__LINE__,i,sum);
  //  first[i]=sum; // may be a bug
  // }


  for (int i = 0; i < Gridsize - 1; i++) {
    unsigned short *val = startflag + i * BLOCK_SIZE;
    for (int j = 0; j < BLOCK_SIZE; j++) {
      if (val[j] == 1) {
        block_flag[i] = 1;
        break;
      }
    }
  }

  int baseindex = (Gridsize - 1) * BLOCK_SIZE;
  unsigned short *val = startflag + baseindex;
  for (int j = 0; j < BLOCK_SIZE && baseindex + j < flagLen; j++) {
    if (val[j] == 1) {
      block_flag[Gridsize - 1] = 1;
      break;
    }
  }
}

#endif