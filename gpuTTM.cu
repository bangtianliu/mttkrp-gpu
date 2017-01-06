#include "gpuTTM.h"
// #include "flag.h"
// #ifdef DOUBLE

// extern __shared__ double M[];

// #else

// extern __shared__ float M[];

// #endif


void test11(stensor data)
{	
	printf("Test:\n");
	for(int i=0;i<data.nnz;i++)
	{
		printf("%d\t%d\t%d\t%f\n", data.i[i],data.j[i], data.k[i],data.val[i]);
	}
}

/**
 * generate lastsum and recahe on each thread
 */

template <typename T, typename type_thread>
__global__ void TTMgpu(const tensor_gpu<T> D_ltensor, 
				  const T * const __restrict__ D_matrix, 
				  const int nRows, 
				  semitensor_gpu<T,type_thread> D_rtensor, 
				  const unsigned int threadlen)
{
	unsigned int tidx=blockIdx.x*blockDim.x + threadIdx.x;
	
	unsigned int tidy=blockIdx.y;

	// int threadlen=D_ltensor.d_threadlen;
	int iterlen=D_ltensor.d_iterlen;
	type_thread mask=1;
	// int rank=D_rtensor.d_nCols;
	
	// extern __shared__ T M[]; // rank*threadlen*block*dim

	
	// int sWidth;
	int nfibs=D_rtensor.d_nfibs;
	// printf("tidx=%d,d_nnz=%d\n", tidx, D_rtensor.d_nnz);
	if(tidx<D_rtensor.d_nnz)
	{
      // if(threadIdx.x==0)printf("%d GPU test\n",__LINE__);
	  type_thread bits=D_rtensor.d_bflags[tidx];
	  // if(tidx==116)printf("####ibits=%x\n", bits);
	  int index_k=D_ltensor.d_k[tidx];
	  T val=D_ltensor.d_val[tidx];
	  T lastsum=val*D_matrix[tidy*nRows+index_k];

	  int d_first=D_rtensor.d_first[tidx];
	  // if(threadIdx.x==124)printf("%d GPU test %f\n",__LINE__,D_matrix[tidy*nRows+index_k]);
	  unsigned short flag= ((mask&bits)==0)? 0:1;
	  unsigned short preFlag;
	  // printf("%d %d\n", tidx,d_first);
	  if(flag==0){
	  	D_rtensor.d_recache[tidy*nfibs+d_first]=lastsum;
	  	d_first++;
	  }	

	
	  for(int i=1;i<threadlen-1;i++)
	  {
	  	preFlag=flag;
	  	flag= (((mask<<i)&bits)==0)? 0 : 1;

	  	// if(threadIdx.x==0)printf("Iteration=%d\n", i);
	  	index_k=D_ltensor.d_k[i*iterlen+tidx];
	  	val=D_ltensor.d_val[i*iterlen+tidx];
	  	lastsum=preFlag*lastsum+val*D_matrix[tidy*nRows+index_k];


        if(flag==0){
        	D_rtensor.d_recache[tidy*nfibs+d_first]=lastsum;
        	d_first++;
        }
	  }

	  preFlag=flag;
	  flag= (((mask<<(threadlen-1))&bits)==0)? 0 : 1;
	  index_k=D_ltensor.d_k[(threadlen-1)*iterlen+tidx];
	  val=D_ltensor.d_val[(threadlen-1)*iterlen+tidx];
	  lastsum=preFlag*lastsum+val*D_matrix[tidy*nRows+index_k];
	  D_rtensor.d_last_partial[tidy*D_rtensor.d_nnz+tidx]=lastsum;
	  if(flag==0){
	  	// if(tidx==116)printf("Hello!!!!!!\n");
	  	 D_rtensor.d_last_partial[tidy*D_rtensor.d_nnz+tidx]=0;
	  	 D_rtensor.d_recache[tidy*nfibs+d_first]=lastsum;
	  }	
	  // if(tidx==116) printf("idx=%d last_partial=%f\n",tidx,D_rtensor.d_last_partial[tidy*D_rtensor.d_nnz+tidx]);
	  // if(tidx==116)printf("idx=%d d_first=%d recache=%f\n",tidx,d_first, D_rtensor.d_recache[d_first]);

	}

	// __synthreads();

}


template <typename T>
__device__ T segscan_warp(T *ptr, unsigned short *hd, const unsigned int d_nnz, const unsigned int idx, const unsigned int idy)
{

	const unsigned int lane=threadIdx.x & 31;
	// if(hd[idx]) hd[idx]=lane;

	unsigned int data_index=idy*d_nnz+idx; // good!
	unsigned short flag=hd[data_index]; //fix
	T value=ptr[data_index];
	
	// if(idx==0) printf("####Test flag idx=%d flag=%d value=%f\n", idx, flag,value);
	__syncthreads();
	// val tmp;
	int warp_size=32;
	if(d_nnz<warp_size)warp_size=d_nnz;
	__syncthreads();
	for(int width=1;width<warp_size;width*=2)
	{
		__syncthreads();
		unsigned int otherflag = __shfl_up(flag,width,32);
		T othervalue = __shfl_up(value,width,32); 
		if(lane>=width){
			// if(idx==3)printf("FF####Test flag idx=%d %d othervalue=%f\n", idx, otherflag,othervalue); 
			value =flag? value : value +othervalue;
			flag = flag | otherflag;
			// if(idx==3)printf("FF####Test flag idx=%d %d value=%f\n", idx, otherflag,value); 
		}
	}
	hd[data_index]=flag;
	ptr[data_index]=value;
	// printf("####Test flag 2 idx=%d flag=%d value=%f\n", idx, flag,value);

	// tmp.value=value;
	// tmp.flag=flag;
	return value;
}

template <typename T> 
__device__ T segscan_block(volatile  T *ptr, 
							 unsigned short *hd, 
							 const int d_nnz, 
							 T *wptr,
							 unsigned short *whd,
							 const unsigned int idx, 
							 const unsigned int idy)
{
	unsigned short warplength = ((blockDim.x-1)/32+1); //d_nnz >> 5;
	
	unsigned short warpid = threadIdx.x >> 5;
	unsigned int warp_first = (blockIdx.x*blockDim.x) + (warpid << 5);  // two different kind of idx
	unsigned int warp_last  = warp_first + 31;





	if(blockIdx.x==gridDim.x-1){
		warplength=(d_nnz-(gridDim.x-1)*blockDim.x-1)/32+1;

		if(warpid==warplength-1)warp_last=d_nnz-1;
	}	


	unsigned short lane_id = threadIdx.x & 31;

	unsigned int dwarp_first = idy*d_nnz + warp_first;

	unsigned int dwarp_last = idy*d_nnz + warp_last;



	bool warp_is_open = (hd[dwarp_first]==0);

	__syncthreads();
	
	T sum = segscan_warp(ptr,hd,d_nnz,idx,idy); // idx, idy

	// if(blockIdx.x==1024)printf("%d idx=%d threadIdx=%d val=%f warp_last=%d \n", __LINE__, idx, threadIdx.x, sum,warp_last);
    
    T warp_total = ptr[dwarp_last];


    unsigned short warp_flag = hd[dwarp_last]!=0;// || !warp_is_open;
    bool will_accumulate = warp_is_open && hd[idy*d_nnz+idx]==0;
    
    __syncthreads();

  
    if(idx == warp_last){
        
    	wptr[warpid]=warp_total;  //share memory for scan in the block
    	whd[warpid]=warp_flag;

    	// if(blockIdx.x==1024)printf("warp_id=%d warp_total=%f warp_flag=%d\n", warpid,warp_total,warp_flag);
    }

	__syncthreads();

	if(warpid ==0 && lane_id<warplength){
		// printf("ffff lane_id=%d",lane_id);
		T value = wptr[lane_id];
		// if(lane_id==2)printf("TEST=%f\n",value);
		unsigned short flag = whd[lane_id];

		for(int i=1;i<=32;i*=2)
		{
			unsigned short otherflag = __shfl_up(flag,i,32);
			T othervalue = __shfl_up(value,i,32);
			// if(blockIdx.x==0)printf("%d i=%d lane_id=%d value=%f othervalue=%f flag=%d otherflag=%d\n", __LINE__, i, lane_id, value, othervalue,flag,otherflag);
			if(lane_id>=i){

				value =flag? value: value +othervalue;
				flag = flag | otherflag; 
			}
		}
		wptr[lane_id]=value;
		// wptr[lane_id+idy*d_nnz]=value;
		whd[lane_id]=flag;
	}

	__syncthreads();

	if(warpid!=0 && will_accumulate){
		sum = sum+wptr[warpid-1]; // fix the bug
		hd[idy*d_nnz+idx]= hd[idy*d_nnz+idx] | whd[warpid-1] ; 
		// hd[idx]=1; // add
	}
	__syncthreads();

	ptr[idy*d_nnz+idx] = sum;

	__syncthreads();

	return sum;
}

template <typename T>
__device__ T propagate( T *last_partial,  //flag
				unsigned short *startflag,
				T preSum,
				int idx,
				int idy,
				int d_nnz)
{

	int data_index=idy*d_nnz+idx;
	// printf("Idx=%d preSum=%f Last=%f\n",idx,preSum,last_partial[data_index] );
	if(startflag[data_index]==0)last_partial[data_index]=last_partial[data_index]+preSum; //fix
	return last_partial[data_index];
}

template <typename T>
__global__ void segmentedscan(T *last_partial, 
							  volatile T *blockSum, 
							  unsigned short *blockflag, 
							  unsigned short *startflag,
							  int d_nnz)
{

	__shared__ T wptr[32];
	__shared__ unsigned short whd[32];

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y;

	if(idx<d_nnz){
		// if(idx==0)printf("%d TEST %d d_nnz=%d\n", __LINE__,gridDim.x,d_nnz);
		T val=segscan_block(last_partial, startflag, d_nnz, wptr,whd, idx, idy);

		// if(blockIdx.x==1024)printf("%d idx=%d threadIdx=%d blockIdx.y=%d val=%f \n", __LINE__, idx, threadIdx.x, blockIdx.y, val);

		if(blockflag[blockIdx.x]==1 || blockIdx.x==0){
			if(threadIdx.x==(blockDim.x-1)){
				// printf("%d %f %f\n", __LINE__,val,blockSum[idy*gridDim.x+blockIdx.x]);
				blockSum[idy*gridDim.x+blockIdx.x]=val;		
			    // if(blockIdx.x==1024)printf("Bangtian Test=%f %f\n", val,blockSum[idy*gridDim.x+blockIdx.x]);
				// printf("%d blockIdx=%d %f %f\n", __LINE__, blockIdx.x, val,blockSum[idy*gridDim.x+blockIdx.x]);
			}
			__syncthreads();


		}
		// if(blockIdx.x==4)printf("TTEST startflag=%d\n", startflag[blockIdx.x*blockDim.x]);
		if(blockIdx.x>0)
		{
			if(startflag[idy*d_nnz+blockIdx.x*blockDim.x]==0){
				T preSum;
				while(blockSum[idy*gridDim.x+blockIdx.x-1]!=blockSum[idy*gridDim.x+blockIdx.x-1]) ;
				__syncthreads();
				preSum=blockSum[idy*gridDim.x+blockIdx.x-1];
				// if(idx==65549) printf("blockIdx=%d preSum=%f\n",blockIdx.x, preSum);
				
				T val=propagate(last_partial,startflag,preSum,idx,idy,d_nnz);

				// if(blockIdx.x==1024)printf("%d idx=%d val=%f\n", __LINE__,threadIdx.x,val);
				__syncthreads();
				// printf("Idx=%d preSum=%f\n",idx, val);
				if(blockflag[blockIdx.x]==0){
					// printf("FFFF%d\n", blockIdx.x);
					if(threadIdx.x==blockDim.x-1){
						blockSum[idy*gridDim.x+blockIdx.x]=val;
						
					}
				}

			}

		}

		// if(idx==0)printf("__LINE__=%d recache\n", __LINE__);
	}
	// T preSum=0;
	
// maybe can use share memory

	// val sum=segscan_warp(last_partial,startflag,d_nnz,idx,idy);
}

template <typename T, typename type_thread>
__global__ void mergeSum(T *last_partial, 
						 T *recache, 
						 int *first, 
						 type_thread *bflags,
						 int d_nnz, 
						 int d_nfibs) // copy a startflag
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y;

	if(idx<d_nnz){
		T lastsum = idx==0 ? 0 : last_partial[idy*d_nnz+idx-1]; 
		type_thread mask=1;
		type_thread bits=bflags[idx];
		type_thread flag;
		int tfirst=first[idx];

		
		for(int i=0; i<8*sizeof(type_thread); i++)
		{
			flag=(mask<<i)&bits;
			flag=(flag==0)? 0 : 1; 
		    // if(idx==65578)printf("i=%d FLAG==%d\n", i,flag);
			if(flag==0){
				// if(idx==65578)printf("Iteration i=%d Last Sum=%f recache=%f\n", i, lastsum,recache[tfirst]);
				recache[idy*d_nfibs+tfirst]=lastsum+recache[idy*d_nfibs+tfirst];
				break;
			}
		}


	}

}

// template <typename T, typename type_thread>
mtype *callTTM(stensor ltensor, mtype *matrix, int nRows, int nCols, semitensor rtensor, type_thread threadtype, int blocksize)
{
	int BLOCK_SIZE=blocksize;

	

	// printf("Into callTTM size=%d\n", sizeof(type_thread));

	// flag<mtype,type_thread> tflag = flag<mtype,type_thread>(rtensor,BLOCK_SIZE);

	int threadlen = sizeof(threadtype)*8;
	transpose_matrix(matrix,nRows,nCols);
	mtype *D_matrix;

	cudaMalloc((void **)&D_matrix,sizeof(mtype)*nRows*nCols);

	cudaMemcpy(D_matrix, matrix, sizeof(mtype)*nRows*nCols,cudaMemcpyHostToDevice);

	transpose_tensor(ltensor,threadlen);

	int nnz=rtensor.nnz;
	printf("TTM: NNZ=%d\n", nnz);
	nnz=(nnz-1)/threadlen+1;
	printf("The number of Threads=%d\n", nnz);
	int rank=nCols;
	// nnz = 32-nnz%32+nnz;
	int Gridsize = (nnz -1)/BLOCK_SIZE +1;


    
	if( nnz<BLOCK_SIZE){
		BLOCK_SIZE=nnz;
		Gridsize=1;
	}
	flag<mtype, type_thread> tflag = flag<mtype,type_thread>(rtensor,BLOCK_SIZE);
	// printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	tensor_gpu<mtype> D_ltensor=tensor_gpu<mtype>(ltensor,threadlen);
	// printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	semitensor_gpu<mtype,type_thread> D_rtensor=semitensor_gpu<mtype,type_thread>(rtensor,tflag,BLOCK_SIZE);

	// printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	// int nnz=D_rtensor.d_nnz;
	// int rank=D_rtensor.d_nCols;



	// printf("Before GPU %d nfibs=%d\n", D_rtensor.d_nnz, D_rtensor.d_nfibs);

	printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	// dim3 grid(Gridsize,rank);
    

    printf("BLOCK:(%d %d) Threads %d in One BLOCK\n", Gridsize,rank,BLOCK_SIZE);
    cudaDeviceSynchronize();

	TTMgpu<<< dim3(Gridsize,rank),BLOCK_SIZE>>>(D_ltensor,
									D_matrix, 
									nRows,
									D_rtensor, 
									threadlen);

	cudaDeviceSynchronize();
	printf("%d, %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	

	segmentedscan<<<dim3(Gridsize,rank), BLOCK_SIZE>>>(D_rtensor.d_last_partial,
														D_rtensor.d_blockSum,
														D_rtensor.d_blockflag,
														D_rtensor.d_startflag,
														D_rtensor.d_nnz);
	cudaDeviceSynchronize();
	printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));

	// printf("d_nfibs=%d\n",D_rtensor.d_nfibs );
	mergeSum<<<dim3(Gridsize,rank),BLOCK_SIZE>>>(D_rtensor.d_last_partial,
												 D_rtensor.d_recache, 
												 D_rtensor.d_first,
												 D_rtensor.d_bflags,
												 D_rtensor.d_nnz,
												 D_rtensor.d_nfibs);


	cudaDeviceSynchronize();
	printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	
	cudaMemcpy((mtype *)D_rtensor.h_result,
		       (mtype *)D_rtensor.d_recache,
		       sizeof(mtype)*D_rtensor.d_nfibs*nCols,
		       cudaMemcpyDeviceToHost);
	printf("TTTEST\n");
	D_ltensor.Free(0);
	D_rtensor.Free(0);
	free(matrix);
	return D_rtensor.h_result;

}





