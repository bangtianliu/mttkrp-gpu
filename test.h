#ifndef _TEST_H
#define _TEST_H

#include "TTM.h"

/**
 * @param data the tensor 
 * @param nnz the number of non-zeros
 */
void test(tensor data, int nnz)
{	
	printf("Test:\n");
	for(int i=0;i<nnz;i++)
	{
		printf("%d\t%d\t%d\t%f\n", data[i].coord[0],data[i].coord[1],data[i].coord[2],data[i].val);
	}
}

void test1(stensor data)
{	
	printf("Test:\n");
	for(int i=0;i<data.nnz;i++)
	{
		printf("%d\t%d\t%d\t%f\n", data.i[i],data.j[i], data.k[i],data.val[i]);
	}
}


void test_TTM(semitensor tensor)
{
	int nfibs=tensor.nfibs;
	int ncols=tensor.R;
	if(nfibs>10)nfibs=10;
	if(ncols>5)ncols=5;
	for(int i=0;i<nfibs;i++)
	{	
		printf("Index i=%d j=%d: ", tensor.i[i],tensor.j[i]);
		for(int j=0;j<ncols;j++)
		{
			printf("%f\t",tensor.val[i*tensor.R+j]);
		}
		printf("\n");
	
    }
}
template <typename T>
void verify(semitensor tensor, T *d_result)
{
	int nfibs=tensor.nfibs;
	int ncols=tensor.R;
	for(int i=0;i<nfibs;i++)
	{
		int index_i=tensor.i[i];
		int index_j=tensor.j[i];
		for(int j=0;j<ncols;j++)
        {
        	if(tensor.val[i*tensor.R+j]!=d_result[j*nfibs+i]){
        		printf("Different at %d %d CPU=%f GPU=%f\n",index_i,index_j,tensor.val[i*tensor.R+j],d_result[j*nfibs+i]);
        	}
        }
	}
}
#endif