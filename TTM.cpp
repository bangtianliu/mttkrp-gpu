#include "TTM.h"

int preprocess(stensor htensor, int **flag, semitensor &result)
{
	int nnz=htensor.nnz;
	*flag=(int *)malloc(sizeof(int)*nnz);
	memset(*flag,0,sizeof(int)*nnz);
	int prev_i=htensor.i[0];
	int prev_j=htensor.j[0];
	(*flag)[0]=1;
	int nfibs=1;
	for(int i=1;i<nnz;i++)
	{
		if(htensor.i[i]!=prev_i||htensor.j[i]!=prev_j){
			(*flag)[i]=1;
			++nfibs;
			prev_i=htensor.i[i];
			prev_j=htensor.j[i];
		}

	}
	result.nfibs=nfibs;
	result.flag=*flag;
	result.nnz=nnz;
	result.i=(int *)malloc(sizeof(int)*nfibs);
	result.j=(int *)malloc(sizeof(int)*nfibs);

	return nfibs;

}


void TTM(stensor htensor, int nfibs, ttype *B, int nCols, semitensor &result)
{
	int nnz=htensor.nnz;
	result.R=nCols;
	ttype *tmp=(ttype *)malloc(sizeof(ttype)*nfibs*nCols);
	memset(tmp,0,sizeof(ttype)*nfibs*nCols);
	int index=-1;
	for (int i = 0; i < nnz; ++i)
	{
		int k=htensor.k[i];
		ttype val=htensor.val[i];
		
		if(result.flag[i]==1){
			++index;
			result.i[index]=htensor.i[i];
			result.j[index]=htensor.j[i];
	    }

	    for(int j=0;j<nCols;j++)
	    {
	    	tmp[index*nCols+j]+=val*B[k*nCols+j];
	    }			/* code */
	}
	result.val=tmp;
}