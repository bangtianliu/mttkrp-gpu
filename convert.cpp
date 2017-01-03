#include "convert.h"

void convert(tensor data, stensor &CPU_tensor,int nnz)
{
	for(int i=0;i<nnz;i++)
	{
		// printf("Intest %d\n", data[i].coord[0]);
		CPU_tensor.i[i]=data[i].coord[0];
		CPU_tensor.j[i]=data[i].coord[1];
		CPU_tensor.k[i]=data[i].coord[2];
		CPU_tensor.val[i]=data[i].val;
		CPU_tensor.nnz=nnz;
	}
}