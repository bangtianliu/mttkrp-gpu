#include "gpuMTTKRP.h"


/**
 * generate lastsum and recahe on each thread
 */

template <typename T, typename type_thread>
__global__ void MTTKRPgpu(const tensor_gpu<T> D_ltensor,
                          const T *const __restrict__ D_Bmatrix,
                          const T *const __restrict__ D_Cmatrix,
                          const unsigned int B_nRows,
                          const unsigned int C_nRows,
                          semitensor_gpu<T, type_thread> D_rtensor,
                          volatile T *blockSum, 
                          const unsigned int threadlen) {

  __shared__ T value[1024];
  __shared__ uint8_t sflag[1024];
  __shared__ T wvalue[32];
  __shared__ uint8_t wflag[32];


  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int tidy = blockIdx.y;

 
  int iterlen = D_ltensor.d_iterlen;
  type_thread mask = 1;

  int nfibs = D_rtensor.d_nfibs;
  // printf("tidx=%d,d_nnz=%d\n", tidx, D_rtensor.d_nnz);
  if (tidx < D_rtensor.d_nnz) {
    // if(threadIdx.x==0)printf("%d GPU test\n",__LINE__);
    type_thread bits = D_rtensor.d_bflags[tidx];
    // if(tidx==116)printf("####ibits=%x\n", bits);
    unsigned int index_j = D_ltensor.d_j[tidx];
    unsigned int index_k = D_ltensor.d_k[tidx];


    T val = D_ltensor.d_val[tidx];

    T lastsum = val * __ldg(&D_Bmatrix[tidy * B_nRows + index_j]) * __ldg(&D_Cmatrix[tidy * C_nRows + index_k]);

    int d_first = D_rtensor.d_first[tidx];
    uint8_t flag = ((mask & bits) == 0) ? 0 : 1;
    uint8_t preFlag;

    if (flag == 0) {
      D_rtensor.d_recache[tidy * nfibs + d_first] = lastsum;
      d_first++;
    }

#pragma unroll
    for (int i = 1; i < threadlen - 1; i++) {
      preFlag = flag;
      flag = (((mask << i)&bits) == 0) ? 0 : 1;

      // if(threadIdx.x==0)printf("Iteration=%d\n", i);
      index_j = D_ltensor.d_j[i * iterlen + tidx];
      index_k = D_ltensor.d_k[i * iterlen + tidx];

      val = D_ltensor.d_val[i * iterlen + tidx];
      lastsum = preFlag * lastsum + val * __ldg(&D_Bmatrix[tidy * B_nRows + index_j]) * __ldg(&D_Cmatrix[tidy * C_nRows + index_k]);


      if (flag == 0) {
        D_rtensor.d_recache[tidy * nfibs + d_first] = lastsum;
        d_first++;
      }
    }

    preFlag = flag;
    flag = (((mask << (threadlen - 1))&bits) == 0) ? 0 : 1;
    index_k = D_ltensor.d_k[(threadlen - 1) * iterlen + tidx];
    val = D_ltensor.d_val[(threadlen - 1) * iterlen + tidx];
    lastsum = preFlag * lastsum + val * __ldg(&D_Bmatrix[tidy * B_nRows + index_j]) * __ldg(&D_Cmatrix[tidy * C_nRows + index_k]);
    value[threadIdx.x]=lastsum;
    // D_rtensor.d_last_partial[tidy * D_rtensor.d_nnz + tidx] = lastsum;
    if (flag == 0) {
      // if(tidx==116)printf("Hello!!!!!!\n");
      value[threadIdx.x]=0;
      // D_rtensor.d_last_partial[tidy * D_rtensor.d_nnz + tidx] = 0;
      D_rtensor.d_recache[tidy * nfibs + d_first] = lastsum;
    }
  

     sflag[threadIdx.x]=D_rtensor.d_startflag[tidx];
     __syncthreads();

     val=segscan_block(value, sflag, D_rtensor.d_nnz, wvalue, wflag, tidx, tidy);

         if (D_rtensor.d_blockflag[blockIdx.x] == 1 || blockIdx.x == 0) {
      if (threadIdx.x == (blockDim.x - 1)) {
        
        // D_rtensor.d_
        blockSum[tidy * gridDim.x + blockIdx.x] = val;
        // __threadfence();
      
      }
    }

  if(sflag[0]==0&&blockIdx.x>0) {
        T preSum;
        while (blockSum[tidy * gridDim.x + blockIdx.x - 1] != blockSum[tidy * gridDim.x + blockIdx.x - 1]) ;
        __syncthreads();



        preSum = blockSum[tidy * gridDim.x + blockIdx.x - 1];
       
    if (D_rtensor.d_blockflag[blockIdx.x] == 0) {
        
          if (threadIdx.x == blockDim.x - 1) {
            blockSum[tidy * gridDim.x + blockIdx.x] = preSum+val;

          }
        }


    T val=propagate(value, sflag, preSum, tidx, tidy, D_rtensor.d_nnz);

       
        __syncthreads();
 
      

      }

          if(blockIdx.x>0) {
      if(threadIdx.x==0){
        while(blockSum[tidy * gridDim.x + blockIdx.x - 1]!=blockSum[tidy * gridDim.x + blockIdx.x - 1]) ;
       
        lastsum = blockSum[tidy * gridDim.x + blockIdx.x - 1];

      }
      else{
        lastsum = value[threadIdx.x-1];

      }

    
    }
    else lastsum = threadIdx.x==0 ? 0 : value[threadIdx.x-1];

    int tfirst =D_rtensor.d_first[tidx];



    for(int i=0;i<threadlen;i++){

      flag = (((mask << i)&bits) == 0) ? 0 : 1;
     
      if(flag==0){

        D_rtensor.d_recache[tidy*nfibs+tfirst] = lastsum + D_rtensor.d_recache[tidy*nfibs+tfirst];
        break;
      }
    }

  }

  // __synthreads();

}


template <typename T>
__device__ T segscan_warp(T *ptr, uint8_t *hd, const unsigned int d_nnz, const unsigned int idx, const unsigned int idy) {

  const unsigned int lane = threadIdx.x & 31;

  uint8_t flag = hd[threadIdx.x]; //fix
  T value = ptr[threadIdx.x];


  int warp_size = 32;

#pragma unroll
  for (int width = 1; width < warp_size; width *= 2) {
    __syncthreads();
    uint8_t otherflag = __shfl_up(flag, width, 32);
    T othervalue = __shfl_up(value, width, 32);
    if (lane >= width) {
    
      value = flag ? value : value + othervalue;
      flag = flag | otherflag;

    }
  }
  hd[threadIdx.x] = flag;
  ptr[threadIdx.x] = value;

  return value;
}

template <typename T>
__device__ T segscan_block(volatile  T *ptr,
                           uint8_t *hd,
                           const int d_nnz,
                           T *wptr,
                           uint8_t *whd,
                           const unsigned int idx,
                           const unsigned int idy) {
  unsigned short warplength = ((blockDim.x - 1) / 32 + 1); //d_nnz >> 5;

  unsigned short warpid = threadIdx.x >> 5;
  unsigned int warp_first =  warpid << 5; // two different kind of idx
  unsigned int warp_last  = warp_first + 31;





  if (blockIdx.x == gridDim.x - 1) {
    warplength = (d_nnz - (gridDim.x - 1) * blockDim.x - 1) / 32 + 1;

    if (warpid == warplength - 1) {
      warp_last = d_nnz-(gridDim.x - 1) * blockDim.x-(warplength-1)*32;
      // warp_last = d_nnz - 1;
    }
  }


  unsigned short lane_id = threadIdx.x & 31;


  bool warp_is_open = (hd[warp_first] == 0);

  __syncthreads();

  T sum = segscan_warp(ptr, hd, d_nnz, idx, idy); // idx, idy

 
  T warp_total = ptr[warp_last];


  uint8_t warp_flag = hd[warp_last] != 0; // || !warp_is_open;
  bool will_accumulate = warp_is_open && hd[threadIdx.x] == 0;

  __syncthreads();


  if (threadIdx.x == warp_last) {

    wptr[warpid] = warp_total; //share memory for scan in the block
    whd[warpid] = warp_flag;

  }

  __syncthreads();

  if (warpid == 0 && lane_id < warplength) {
    
    T value = wptr[lane_id];
   
    uint8_t flag = whd[lane_id];

#pragma unroll
    for (int i = 1; i <= 32; i *= 2) {
      uint8_t otherflag = __shfl_up(flag, i, 32);
      T othervalue = __shfl_up(value, i, 32);
    
      if (lane_id >= i) {

        value = flag ? value : value + othervalue;
        flag = flag | otherflag;
      }
    }
    wptr[lane_id] = value;

    whd[lane_id] = flag;
  }

  __syncthreads();

  if (warpid != 0 && will_accumulate) {
    sum = sum + wptr[warpid - 1]; 
    hd[threadIdx.x] = hd[threadIdx.x] | whd[warpid - 1] ;

  }
  __syncthreads();

  ptr[threadIdx.x] = sum;

  __syncthreads();

  return sum;
}

template <typename T>
__device__ T propagate( T *last_partial,  //flag
                        uint8_t*startflag,
                        T preSum,
                        int idx,
                        int idy,
                        int d_nnz) {


  if (startflag[threadIdx.x] == 0) {
    last_partial[threadIdx.x] = last_partial[threadIdx.x] + preSum;  //fix
  }
  return last_partial[threadIdx.x];
}

template <typename T>
__global__ void segmentedscan(T *last_partial,
                              volatile T *blockSum,
                              uint8_t *blockflag,
                              uint8_t *startflag,
                              int d_nnz) {

  __shared__ T wptr[32];
  __shared__ uint8_t whd[32];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y;

  if (idx < d_nnz) {

    T val = segscan_block(last_partial, 
    						startflag, 
    						d_nnz, 
    						wptr, 
    						whd, 
    						idx, 
    						idy);



    if (blockflag[blockIdx.x] == 1 || blockIdx.x == 0) {
      if (threadIdx.x == (blockDim.x - 1)) {

        blockSum[idy * gridDim.x + blockIdx.x] = val;

      }
      __syncthreads();


    }

    if (blockIdx.x > 0) {
      if (startflag[idy * d_nnz + blockIdx.x * blockDim.x] == 0) {
        T preSum;
        while (blockSum[idy * gridDim.x + blockIdx.x - 1] != blockSum[idy * gridDim.x + blockIdx.x - 1]) ;
        __syncthreads();
        preSum = blockSum[idy * gridDim.x + blockIdx.x - 1];


        T val = propagate(last_partial, startflag, preSum, idx, idy, d_nnz);

        __syncthreads();
      
        if (blockflag[blockIdx.x] == 0) {
       
          if (threadIdx.x == blockDim.x - 1) {
            blockSum[idy * gridDim.x + blockIdx.x] = val;

          }
        }

      }

    }

  }

}

template <typename T, typename type_thread>
__global__ void mergeSum(T *last_partial,
                         T *recache,
                         int *first,
                         type_thread *bflags,
                         int d_nnz,
                         int d_nfibs) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y;

  if (idx < d_nnz) {
    T lastsum = idx == 0 ? 0 : last_partial[idy * d_nnz + idx - 1];
    type_thread mask = 1;
    type_thread bits = bflags[idx];
    type_thread flag;
    int tfirst = first[idx];

#pragma unroll
    for (int i = 0; i < 8 * sizeof(type_thread); i++) {
      flag = (mask << i)&bits;
      flag = (flag == 0) ? 0 : 1;

      if (flag == 0) {
       
        recache[idy * d_nfibs + tfirst] = lastsum + recache[idy * d_nfibs + tfirst];
        break;
      }
    }


  }

}

mtype *callTTM(stensor ltensor, mtype *Bmatrix, mtype *Cmatrix, int B_nRows, int C_nRows, int nCols, semitensor rtensor, type_thread threadtype, int blocksize) {
  int BLOCK_SIZE = blocksize;



  int threadlen = sizeof(threadtype) * 8;
  transpose_matrix(Bmatrix, B_nRows, nCols);
  transpose_matrix(Cmatrix, C_nRows, nCols);

  mtype *D_Bmatrix;
  mtype *D_Cmatrix;

  cudaMalloc((void **)&D_Bmatrix, sizeof(mtype)*B_nRows * nCols);
  cudaMalloc((void **)&D_Cmatrix, sizeof(mtype)*C_nRows * nCols);


  cudaMemcpy(D_Bmatrix, Bmatrix, sizeof(mtype)*B_nRows * nCols, cudaMemcpyHostToDevice);
  cudaMemcpy(D_Cmatrix, Cmatrix, sizeof(mtype)*C_nRows * nCols, cudaMemcpyHostToDevice);


  transpose_tensor(ltensor, threadlen);

  int nnz = rtensor.nnz;
  printf("MTTKRP gpu: NNZ=%d\n", nnz);

  printf("The size of each thread %d\n", sizeof(type_thread));

  nnz = (nnz - 1) / threadlen + 1;
  printf("The number of Threads=%d\n", nnz);
  int rank = nCols;

  int Gridsize = (nnz - 1) / BLOCK_SIZE + 1;



  if ( nnz < BLOCK_SIZE) {
    BLOCK_SIZE = nnz;
    Gridsize = 1;
  }
  flag<mtype, type_thread> tflag = flag<mtype, type_thread>(rtensor, BLOCK_SIZE);
  tensor_gpu<mtype> D_ltensor = tensor_gpu<mtype>(ltensor, threadlen);

  semitensor_gpu<mtype, type_thread> D_rtensor = semitensor_gpu<mtype, type_thread>(rtensor, tflag, BLOCK_SIZE);


  printf("%d %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));



  printf("BLOCK:(%d %d) Threads %d in One BLOCK\n", Gridsize, rank, BLOCK_SIZE);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float et = 0;
  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);
  MTTKRPgpu <<< dim3(Gridsize, rank), BLOCK_SIZE>>>(D_ltensor,
                                                    D_Bmatrix,
                                                    D_Cmatrix,
                                                    B_nRows,
                                                    C_nRows,
                                                    D_rtensor,
                                                    D_rtensor.d_blockSum,
                                                    threadlen);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&et, start, stop);

  printf("GPU TTM Time: %f s\n", et / 1000.0);

  cudaDeviceSynchronize();
  printf("%d %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy((mtype *)D_rtensor.h_result,
             (mtype *)D_rtensor.d_recache,
             sizeof(mtype)*D_rtensor.d_nfibs * nCols,
             cudaMemcpyDeviceToHost);
  printf("TTTEST\n");
  D_ltensor.Free(0);
  D_rtensor.Free(0);
  free(Bmatrix);
  free(Cmatrix);
  return D_rtensor.h_result;

}





