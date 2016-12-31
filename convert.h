#ifndef _CONVERT_H
#define _CONVERT_H
#include "readtensor.h"
#include <stdlib.h>
#include <malloc.h>

template <typename T>
struct soa_tensor{
public:
	int *i;
	int *j;
	int *k;
	T *val;
	int nnz;
	soa_tensor(int length):nnz(length){
		i=(int *)malloc(sizeof(int)*length);
		j=(int *)malloc(sizeof(int)*length);
		k=(int *)malloc(sizeof(int)*length);
		val=(T *)malloc(sizeof(T)*length);
	}
};

#ifdef DOUBLE
typedef soa_tensor<double> stensor;
// typedef soa_tensor<double> *tensor;
#else
typedef soa_tensor<float> stensor;
#endif

void convert(tensor data, stensor &CPU_tensor, int nnz);
#endif