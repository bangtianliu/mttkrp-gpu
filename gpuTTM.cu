#include "gpuTTM.h"
// #include "flag.h"
#ifdef DOUBLE

extern __shared__ double M[];

#else

extern __shared__ float M[];

#endif


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
	  if(tidx==4)printf("####ibits=%x\n", bits);
	  int index_k=D_ltensor.d_k[tidx];
	  T val=D_ltensor.d_val[tidx];
	  T lastsum=val*D_matrix[tidy*nRows+index_k];
	  // last_partial[tidy*D_rtensor.d_nnz+tidx]=val*D_matrix[tidy*nRows+index_k];
	  int d_first=D_rtensor.d_first[tidx];
	  // if(threadIdx.x==0)printf("%d GPU test %f\n",__LINE__,val);
	  unsigned short flag= ((mask&bits)==0)? 0:1;
	  unsigned short preFlag;
	  printf("%d %d\n", tidx,d_first);
	  if(flag==0){
	  	D_rtensor.d_recache[tidy*nfibs+d_first]=lastsum;
	  	d_first++;
	  }	
	  	  	if(tidx==4){
		printf("%d TEST recache d_first=%d %f\n", __LINE__, d_first, D_rtensor.d_recache[d_first-1]);
		}
	  // if(threadIdx.x==0)printf("%d GPU test %f\n",__LINE__,lastsum);
	  // 	  printf("recache %f\n", D_rtensor.d_recache[0]);
	  // printf("recache %f\n", D_rtensor.d_recache[1]);

	  // last_partial[tidy*D_rtensor.d_nnz+tidx]=
	  // sWidth=threadlen*blockDim.x;	
	  for(int i=1;i<threadlen-1;i++)
	  {
	  	preFlag=flag;
	  	flag= (((mask<<i)&bits)==0)? 0 : 1;

	  	// if(threadIdx.x==0)printf("Iteration=%d\n", i);
	  	index_k=D_ltensor.d_k[i*iterlen+tidx];
	  	val=D_ltensor.d_val[i*iterlen+tidx];
	  	lastsum=preFlag*lastsum+val*D_matrix[tidy*nRows+index_k];
        // last_partial[tidy*D_rtensor.d_nnz+tidx]=tmp;
	  	if(tidx==0){
	  		printf("Test lastsum=%f preFlag=%d Matrix=%f index_k=%d val=%f\n", lastsum,preFlag,D_matrix[tidy*nRows+index_k],index_k,val);
	  	}
        if(flag==0){
        	D_rtensor.d_recache[d_first]=lastsum;
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
	  	 D_rtensor.d_last_partial[tidy*D_rtensor.d_nnz+tidx]=0;
	  	 D_rtensor.d_recache[d_first]=lastsum;
	  }	
	  printf("idx=%d last_partial=%f\n",tidx,D_rtensor.d_last_partial[tidy*D_rtensor.d_nnz+tidx]);
	  printf("idx=%d d_first=%d last_partial=%f\n",tidx,d_first, D_rtensor.d_recache[d_first]);

	}

	// __synthreads();

}


template <typename T>
__device__ T segscan_warp(T *ptr, unsigned short *hd, const int d_nnz, const unsigned int idx, const unsigned int idy)
{
	const unsigned int lane=idx & 31;
	// if(hd[idx]) hd[idx]=lane;

	int data_index=idy*d_nnz+idx; // good!
	unsigned int flag=hd[idx];
	
	T value=ptr[data_index];
	printf("Test flag %d %d %f\n", idx, flag,value);
	// val tmp;
	int warp_size=32;
	if(d_nnz<warp_size)warp_size=d_nnz;
	for(int width=1;width<warp_size;width*=2)
	{
		unsigned int otherflag = __shfl_up(flag,width,32);
		T othervalue = __shfl_up(value,width,32); 
		if(lane>=width){
			
			printf("####Test flag idx=%d %d othervalue=%f\n", idx, otherflag,othervalue); 
			value =flag? value : value +othervalue;
			flag = flag | otherflag;
			printf("####Test flag idx=%d %d value=%f\n", idx, otherflag,value); 
		}
	}
	hd[idx]=flag;
	ptr[data_index]=value;

	// tmp.value=value;
	// tmp.flag=flag;
	return value;
}

template <typename T> 
__device__ T segscan_block(T *ptr, 
							 unsigned short *hd, 
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
	bool warp_is_open = (hd[warp_first]==0);

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
__global__ void segmentedscan(T *last_partial, 
							  T *blockSum, 
							  unsigned short *blockflag, 
							  unsigned short *startflag,
							  unsigned short *startflag_backup, 
							  int d_nnz)
{

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y;

	if(idx<d_nnz){
		// printf("%d TEST %d d_nnz=%d\n", __LINE__,gridDim.x,d_nnz);
		T val=segscan_block(last_partial, startflag, d_nnz, idx, idy);
		printf("#####%d idx=%d %f\n####", __LINE__,idx,val);
		// printf("%d TEST\n", __LINE__);
		if(blockflag[blockIdx.x]==1 || blockIdx.x==0){
			if(threadIdx.x==(blockDim.x-1)){
				printf("%d %f %f\n", __LINE__,val,blockSum[idy*gridDim.x+blockIdx.x]);
				blockSum[idy*gridDim.x+blockIdx.x]=val;		
				printf("%d %f %f\n", __LINE__,val,blockSum[idy*gridDim.x+blockIdx.x]);
			}
			__syncthreads();
		}

		if(blockIdx.x>0)
		{
			if(startflag[blockIdx.x*blockDim.x]==1){
				T val=segscan_block(last_partial, startflag_backup,d_nnz,idx,idy);
				if(blockSum[blockIdx.x]!=blockSum[blockIdx.x]){
					if(threadIdx.x==blockDim.x-1)
						blockSum[idy*gridDim.x+blockIdx.x]=val;
				}
			}
			else {
				printf("into %d\n", idx);
			    
				if(threadIdx.x==0){
					while(blockSum[blockIdx.x-1]!=blockSum[blockIdx.x-1]) ;
					if(startflag[idx]==0){
						last_partial[idy*d_nnz+idx]+=blockSum[idy*gridDim.x+blockIdx.x-1];
					}
				}
				__syncthreads();
				T val=segscan_block(last_partial, startflag_backup,d_nnz,idx,idy);
				if(blockSum[blockIdx.x]!=blockSum[blockIdx.x]){
					if(threadIdx.x==blockDim.x-1)
						blockSum[idy*gridDim.x+blockIdx.x]=val;
				}

			}

		}
	}
	// T preSum=0;
	
// maybe can use share memory

	// val sum=segscan_warp(last_partial,startflag,d_nnz,idx,idy);
}

template <typename T, typename type_thread>
__global__ void mergeSum(T *last_partial, T *recache, int *first, type_thread *bflags,int d_nnz, int d_nfibs) // copy a startflag
{
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y;

	if(idx<d_nnz){
		T lastsum = idx==0 ? 0 : last_partial[idy*d_nnz+idx-1]; 
		type_thread mask=1;
		type_thread bits=bflags[idx];
		type_thread flag;
		int tfirst=first[idx];

		// if(idx==4){
		// 	printf("Last Sum=%f recache=%f\n", lastsum,recache[tfirst]);
		// }
		if(idx==0){
				if(idx==0){
			printf("%d TEST recache %f\n", __LINE__, recache[0]);
			printf("%d TEST recache %f\n", __LINE__, recache[1]);
			printf("%d TEST recache %f\n", __LINE__, recache[2]);
			printf("%d TEST recache %f\n", __LINE__, recache[3]);
			// printf("%d TEST recache %f\n", __LINE__, recache[4]);
		}
		}
		for(int i=0; i<8*sizeof(type_thread); i++)
		{
			flag=(mask<<i)&bits;
			flag=(flag==0)? 0 : 1; 
			if(flag==0){
				recache[tfirst]=lastsum+recache[tfirst];
				break;
			}
		}
		if(idx==0){
			printf("%d TEST recache %f\n", __LINE__, recache[0]);
			printf("%d TEST recache %f\n", __LINE__, recache[1]);
			printf("%d TEST recache %f\n", __LINE__, recache[2]);
			printf("%d TEST recache %f\n", __LINE__, recache[3]);
			// printf("%d TEST recache %f\n", __LINE__, recache[4]);
		}

	}

}

// template <typename T, typename type_thread>
mtype *callTTM(stensor ltensor, mtype *matrix, int nRows, int nCols, semitensor rtensor, type_thread threadtype, int blocksize)
{
	int BLOCK_SIZE=blocksize;

	

	printf("Into callTTM size=%d\n", sizeof(type_thread));

	// flag<mtype,type_thread> tflag = flag<mtype,type_thread>(rtensor,BLOCK_SIZE);

	int threadlen = sizeof(threadtype)*8;
	transpose_matrix(matrix,nRows,nCols);
	mtype *D_matrix;

	cudaMalloc((void **)&D_matrix,sizeof(mtype)*nRows*nCols);

	cudaMemcpy(D_matrix, matrix, sizeof(mtype)*nRows*nCols,cudaMemcpyHostToDevice);

	transpose_tensor(ltensor,threadlen);

	int nnz=rtensor.nnz;
	nnz=(nnz-1)/threadlen+1;
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
    cudaDeviceSynchronize();
	TTMgpu<<< dim3(Gridsize,rank),BLOCK_SIZE, sizeof(mtype)*BLOCK_SIZE*threadlen>>>(D_ltensor,
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
														D_rtensor.d_startflag_backup,
														D_rtensor.d_nnz);
	cudaDeviceSynchronize();
	printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));

	printf("d_nfibs=%d\n",D_rtensor.d_nfibs );
	mergeSum<<<dim3(Gridsize,rank),BLOCK_SIZE>>>(D_rtensor.d_last_partial,
												 D_rtensor.d_recache, 
												 D_rtensor.d_first,
												 D_rtensor.d_blockflag,
												 D_rtensor.d_nnz,
												 D_rtensor.d_nfibs);


	cudaDeviceSynchronize();
	printf("%d %s\n",__LINE__, cudaGetErrorString(cudaGetLastError()));
	
	cudaMemcpy((mtype *)D_rtensor.h_result,
		       (mtype *)D_rtensor.d_recache,
		       sizeof(mtype)*D_rtensor.d_nfibs*nCols,
		       cudaMemcpyDeviceToHost);

	D_ltensor.Free(0);
	D_rtensor.Free(0);
	return D_rtensor.h_result;

}





