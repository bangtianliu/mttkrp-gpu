#ifndef _MATRIX_H
#define _MATRIX_H
#include <stdlib.h>
#include <cstring>
#include <string.h>

template<typename T> 
void genMatrix(T **data, int nRows, int nCols)
{
   *data=(T *)malloc(sizeof(T)*nRows*nCols);
}

template<typename T>
void randomFill(T *data, int nRows,int nCols)
{
	for(int i=0;i<nRows;i++)
		for(int j=0;j<nCols;j++)
		{
			data[i*nCols+j] = i*1.0;
		}
}
template<typename T>
void freeMatrix(T **data)
{
	if(*data!=NULL)
	{
		free(*data);
		(*data)=NULL;
	}
}



#endif