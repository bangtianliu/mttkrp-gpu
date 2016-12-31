#include "flag.h"


template<typename T>
flag<T>::flag(semiTensor<T> tensor)
{
	int threadLen=sizeof(T)*8;
	int nnz=tensor.nnz;

	int flagLen=(nnz-1)/threadLen+1;

	flagLen = 32-flagLen%32+flagLen
	int BLOCK_SIZE=tensor.BLOCK_SIZE;

	int Gridsize=(flagLen-1)/BLOCK_SIZE+1;

	cflag=(T *)malloc(sizeof(T)*flagLen);
	bit_flag=(T *)malloc(sizeof(T)*flagLen);
	first=(int *)malloc(sizeof(int)*flagLen);
	startflag=(unsigned short *)malloc(sizeof(unsigned short)*flagLen);
	block_flag=(unsigned short *)malloc(sizeof(unsigned short)*Gridsize);

	memset(bit_flag,-1,sizeof(T)*flagLen);
	memset(cflag,0,sizeof(T)*flagLen);
	memset(startflag,0,sizeof(unsigned short)*flagLen);
	memset(block_flag,0,sizeof(unsigned short)*Gridsize);
	for(int i=0;i<flagLen;i++)
	{
		// T ibits;
		T bits=0;
		for(int j=0;j<threadLen && (i*threadLen+j)<nnz;j++)
		{
			unsigned int elem=tensor.flag[i*threadLen+j];
			bits+=(elem<<j);
		}
		cflag[i]=bits;

	}

	for(int i=0;i<flagLen-1;i++)
	{
		T ibits=0;
		for(int j=0;j<threadLen;j++)
		{
			unsigned int nextelem=tensor.flag[i*threadLen+j+1];
			if(nextelem==1)ibits+=(1<<j);	
		}
		bit_flag[i]-=ibits;				
	}

	T ibits=0;
	for(int j=0;j<threadLen;j++)
	{
		int index = (flagLen-1)*threadLen+j;
		unsigned int nextelem;
		if(index<nnz-1){
			nextelem=tensor.flag[index];
			if(nextelem==1)ibits+=(1<<j);
		}
		if(index==nnz-1){
			;
			// ibits+=(1<<j);			
		}
		if(index>nnz-1&&j<threadLen-1){
			;
		}
		if(j=threadLen-1){
			ibits+=(1<<j);
		}

	}
	bit_flag[flagLen-1]-=ibits;


    for(int i=0; i<flagLen; i++)
    {
    	if(bit_flag[i]!=numeric_limits<T>::max()){
    		startflag[i]=1;
    	}
    }

	first[0]=0; // first result entry on each thread
	for(int i=1;i<flagLen;i++)
	{	
		int sum=0;
		for(int j=0;j<threadLen&&(i*threadLen+j)<nnz;j++)
		{
			if(flag[(i-1)*threadLen+j]==1){
				++sum;
			}
		}
		first[i]=sum-1;
	}


	for(int i=0;i<Gridsize-1;i++)
	{
		T *val=startflag+i*BLOCK_SIZE;
		for(int j=0;j<BLOCK_SIZE;j++)
		{
			if(val[j]==1){
				blockflag[i]=1;
				break;
			}
		}
	}

	int baseindex = (Gridsize-1)*BLOCK_SIZE;
	T *val=startflag+baseindex;
	for(int j=0;j<BLOCK_SIZE&&baseindex+j<flagLen;j++){
			if(val[j]==1){
				blockflag[Gridsize-1]=1;
				break;
			}
	}
}