#include "gpuTTM.h"

#ifdef DOUBLE

extern __shared__ double M[];

#else

extern __shared__ float M[];

#endif


/**
 * generate lastsum and recahe on each thread
 */
template <typename T, typename type_thread>
__device__ void threadcan(T *M, type_thread bits, int tidx, int rank, int threadlen, int sWidth, int nfibs, int *first, T *last_partial, T* recache)
{
	type_thread mask=1;

	unsigned short flag = (mask<<0)&bits;
	unsigned short preFlag;
	int d_first=first[tidx];
	for(int j=0;j<rank;j++)
	{
		last_partial[j*sWidth+tidx]=M[j*sWidth];
		flag=(mask)&bits;
		if(flag==0){
			recache[j*nfibs+d_first]=M[j*sWidth];
			d_first++;
		};
	}

	for(int i=1;i<threadlen;i++)
	{
		
		preFlag = flag;
		flag=(mask<<i)&bits;

		if(flag!=0){
			for(int j=0;j<rank;j++)
			{
				last_partial[j*sWidth]=preFlag*last_partial[j*sWidth]+M[j*sWidth+i]; //maybe can used memory			 
			}
		}
		else 
		{	
			if(i!=threadlen-1){

				for(int j=0;j<rank;j++)
			    {
					T tmp=preFlag*last_partial[j*sWidth]+M[j*sWidth+i]; //maybe can used memory			 
					recache[j*nfibs+d_first]=tmp;
					last_partial[j*sWidth]=tmp;
					d_first++;
				}	
			}
			else{
				for(int j=0;j<rank;j++)
			    {
					T tmp=preFlag*last_partial[j*sWidth]+M[j*sWidth+i]; //maybe can used memory			 
					recache[j*nfibs+d_first]=tmp;
					last_partial[j*sWidth]=0;
					d_first++;
				}
			}
	    }
	}
}

template <typename T, typename type_thread>
__global__ void TTMgpu(const tensor_gpu<T> D_ltensor, 
				  T *D_matrix, 
				  const int nRows, 
				  const int nCols, 
				  semitensor_gpu<T,type_thread> &D_rtensor, 
				  const unsigned int threadlen)
{
	unsigned int tidx=blockIdx.x*blockDim.x + threadIdx.x;
	// int threadlen=D_ltensor.d_threadlen;
	int iterlen=D_ltensor.d_iterlen;
	int rank=D_rtensor.d_nCols;
	type_thread bits=D_rtensor.d_bflags[tidx];
	extern __shared__ T M[]; // rank*threadlen*block*dim

	int sWidth;
	int nfibs=D_rtensor.nfibs;
	if(tidx<D_ltensor.d_nnz)
	{			
	  sWidth=threadlen*blockDim.x;	
	  for(int i=0;i<threadlen;i++)
	  {
	  	int index_k=D_ltensor.d_k[i*iterlen+tidx];
	  	float val=D_ltensor.d_val[i*iterlen+tidx];
	  	for(int j=0;j<rank;j++)
	  	{
	  		M[j*sWidth+threadIdx.x*threadlen+i]=val*D_matrix[j*nRows+index_k];
	  	}
	  }

	  __syncthreads();

	  threadcan(M,bits,tidx,rank,threadlen,sWidth,nfibs,
	  			D_rtensor.d_first,
	  			D_rtensor.d_last_partial,
	  			D_rtensor.d_recache);
	}

	

	// __synthreads();

}


template <typename T>
__device__ T segscan_warp(volatile T *ptr, volatile unsigned short *hd, const int d_nnz, const unsigned int idx, const unsigned int idy)
{
	const unsigned int lane=idx & 31;
	// if(hd[idx]) hd[idx]=lane;

	int data_index=idy*d_nnz+idx; // good!
	unsigned short flag=hd[idx];
	T value=ptr[data_index];
	// val tmp;
	for(int width=1;width<32;width*=2)
	{
		if(lane>=width){
			unsigned short otherflag = __shfl_up(flag,width);
			T othervalue = __shfl_up(value,width);  
			value =flag? value : value +othervalue;
			flag = flag | otherflag;
		}
	}
	hd[idx]=flag;
	ptr[data_index]=value;

	// tmp.value=value;
	// tmp.flag=flag;
	return value;
}

template <typename T> 
__device__ T segscan_block(volatile T *ptr, 
							 volatile unsigned short *hd, 
							 const int d_nnz, 
							 const unsigned int idx, 
							 const unsigned int idy)
{
	unsigned short warplength = d_nnz >> 5;

	unsigned short warpid = threadIdx.x >> 5;
	unsigned short warp_first = (blockIdx.x*blockDim.x) + warpid << 5;  // two different kind of idx
	unsigned short warp_last  = (blockIdx.x*blockDim.x) + warpid + 31;

	unsigned short lane_id = threadIdx.x & 31;

	unsigned short dwarpid = idy*warplength+warpid;
	unsigned short dwarp_first = dwarpid << 5;
	unsigned short dwarp_last = dwarp_first + 31;

	// int idx=blockIdx.x*blockDim.x+threadIdx.x;
	// int idy=threadIdx.y;
	bool warp_is_open = (hd[warpid]==0);

	__syncthreads();
	
	T sum = segscan_warp(ptr,hd,d_nnz,idx,idy); // idx, idy
    
    T warp_total = ptr[dwarp_last];

    unsigned short warp_flag = hd[warp_last]!=0 || !warp_is_open;
    bool will_accumulate = warp_is_open && hd[idx]==0;
    
    __syncthreads();

    

    if(idx == warp_last){
    	ptr[dwarpid]=warp_total;
    	hd[warpid]=warp_flag;
    }

	__syncthreads();

	if(warpid ==0 && lane_id<(blockDim.x/32)){
		T value = ptr[lane_id+idy*d_nnz];
		unsigned short flag = hd[lane_id];
		for(int i=1;i<=(blockDim.x/32);i*=2)
		{
			if(lane_id>=i){
				unsigned short otherflag = __shfl_up(flag,i);
				T othervalue = __shfl_up(value,i);
				value =flag? value: value +othervalue;
				flag = flag | otherflag; 
			}
		}
		ptr[lane_id+idy*d_nnz]=value;
		hd[lane_id]=flag;
	}

	__syncthreads();

	if(warpid!=0 && will_accumulate)
		sum = sum+ptr[dwarpid-1]; // fix the bug
	__syncthreads();

	ptr[idy*d_nnz+idx] = sum;

	__syncthreads();

	return sum;
}

template <typename T>
__global__ void segmentedscan(T *last_partial, T *blockSum, unsigned short *blockflag, unsigned short *startflag, int d_nnz)
{

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=threadIdx.y;
	if(idx<d_nnz){
		if(blockIdx.x==0){
			
			T val=segscan_block(last_partial, startflag, d_nnz, idx, idy);
			if(threadIdx.x==blockDim.x-1){
				blockSum[idy*gridDim.x+blockIdx.x]=val;
			}

		}

		else{

			while(blockSum[blockIdx.x-1]==blockSum[blockIdx.x-1]) ;
			if(threadIdx.x==0){
				if(startflag[idx]==0){
					last_partial[idy*d_nnz+idx]+=blockSum[idy*gridDim.x+blockIdx.x-1];
				}
			}
			T val=segscan_block(last_partial,startflag,d_nnz,idx,idy);
			if(threadIdx.x==blockDim.x-1){
				blockSum[idy*gridDim.x+blockIdx.x]=val;
			}
		}	

	}
	// T preSum=0;
	
// maybe can use share memory

	// val sum=segscan_warp(last_partial,startflag,d_nnz,idx,idy);
}

template <typename T, typename type_thread>
__global__ void mergeSum(T *last_partial, T * recache, int *first, type_thread *bflags,int d_nnz) // copy a startflag
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=threadIdx.y;

	if(idx<d_nnz){
		T lastsum = idx==0 ? 0 : last_partial[idy*d_nnz+idx-1]; 
		type_thread mask=1;
		type_thread bits=bflags[idx];
		unsigned short flag;
		int tfirst=first[idx];
		for(int i=0; i<8*sizeof(type_thread); i++)
		{
			flag=(mask<<i)&bits;
			if(flag==0){
				recache[tfirst]=lastsum+recache[tfirst];
				break;
			}
		}
	}

}



