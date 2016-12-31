#ifndef _READ_H
#define _READ_H


template<typename T>
struct element
{
	int coord[3];
	T val;
};

#ifdef DOUBLE
typedef element<double> item;
typedef element<double> *tensor;
#else
typedef element<float> item;
typedef element<float> *tensor;
#endif

int precess(int &dim_i,int &dim_j, int &dim_k, char *file);

void tensor_malloc(tensor *data, int nnz);
void tensor_free(tensor data);

int readtensor(tensor data, char *file);

#endif