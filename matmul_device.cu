#include <cassert>
#include <cuda_runtime.h>
#include "matmul_device.cuh"


/*
 * Read TODO items below
 */

#define TILE_SIZE 32

// there are 32 way bank conflicts when loading and writing to global memory here.
// not coalesced.
// almost every single load to i,j encounters a memory bank conflict.
// There are memory bank conflicts with many of the stores as well.

__global__
void naiveMatmul(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float acc = 0;
    for (int k=0; k<n; k++) {
	acc += a[i*n+k] * b[k*n+j];
    }
    c[i*n+j] = acc;
}


//The idea here is to interchange loops so that the inner most look becomes cache friendly
//Instead of assigning threads to C, threads are assigned to A to get cache coherence
//It can be seen by looking at the inner loop. There are no jumps in memory.
__global__
void cacheMatmul(float *a, float *b, float *c, int n) {
    // TODO: replace this function with cache friendly version
    //  This is the best we have. We reordered the loops.
   // We hit a bottleneck with atomicAdd().   
   // There are memory bank conflicts when loading to i,j

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float r = a[i*n+j];
   #pragma unroll 
   for (int k=0; k<n; k++) {
//atomic operation of adding up c because many threads will be reading and writing at the same time.
//Very inefficent. Should be replaced by parallel reduction.
	atomicAdd(&c[n*i+k], r*b[k+n*j]);
    }
}
// */


//Implementation of Cache coherent version in shared memory
//The scheme gives a speed up but not as much as the one presented later
//Its because of atomic operation and inabliity to use shared memory for C matrix efficiently.
//If were to be implemented efficiently, atomic operation for writing C in cache memory should 
// be performed. That would be an inefficient model.

//__global__
//void sharedMatmul(float *a, float *b, float *c, int n) {
//    // TODO: replace this function with optimized code using
//    // shared memory
//    const int TS= 32;
//    const int numhops= n/TS;
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    float r = a[i*n+j];
//    __shared__ float tileb[TS][TS];
//
//    for(int hop=0; hop<numhops; hop++){
//    	tileb[threadIdx.y][threadIdx.x]=b[j*n+threadIdx.x+hop*TS];
// 
//    	for (int k=0; k<TS; k+=1) {
//		atomicAdd(&c[n*i+k+hop*TS], r*tileb[threadIdx.y][k]);
//
//    }
//}
//}


__global__
void sharedMatmul(float *a, float *b, float *c, int n) {
    // TODO: replace this function with optimized code using
    // shared memory
    // This is out optimized version of Matrix Multiplication, using 
    // shared memory. The code breaks up the matrices into blocks of 
    // size [TILE_SIZE][TILE_SIZE], with the most efficient being
    // [32][32]. 
    // I get no bank conflicts when loading and storing from and to global memory,
    // but I get 32 way bank conflicts with tileA when loading each element from 
    // shared memory. This occurs when I multiply elements of A with elements of B 
    // to sum for elements of C.
    //
    // I can pad the tileA matrix, and when I do, I don't have any memory bank conflicts,
    // but the efficiency is worse.
	
    const int thread_loc = threadIdx.y*n+threadIdx.x;
    const int A_block_loc = n*TILE_SIZE*blockIdx.y;
    const int B_block_loc = TILE_SIZE*blockIdx.x;
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    float tempC = 0;
    for(int k = 0; k < (n/TILE_SIZE); k++){	
//  read from global memory, and coalesced writes to shared memory.
//  There are no memory bank conflicts here.
	tileA[threadIdx.y][threadIdx.x] = a[thread_loc + A_block_loc+TILE_SIZE*k];
	tileB[threadIdx.y][threadIdx.x] = b[thread_loc + B_block_loc+n*TILE_SIZE*k];
	__syncthreads();
	for( int index = 0; index < TILE_SIZE; index++){
// read from shared memory.
// tileB is coalesced, but tile A has 32way bank conflicts when loading.
// Padding the matrix removes the bank conflicts, but efficiency is becomes worse.
	tempC += tileA[threadIdx.y][index]*tileB[index][threadIdx.x];
	}
	__syncthreads();
   }
// coalesced write to global memory. no memory bank conflicts.
    c[thread_loc + A_block_loc+B_block_loc ] =  tempC;
}
// */

void cudaMatmul(float *a, float *b, float *c, int n, MatmulImplementation type)
{
    // TODO: play with the gridSize and blockSize to find the best one
    // For the SHARED version, a square block size of 32 is most efficient.
    // Other sizes that divide 64 work, but more memory bank conflicts are introduced.

    if (type == NAIVE) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        naiveMatmul<<<gridSize, blockSize>>>(a,b,c,n);
    }
    else if (type == CACHE) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        cacheMatmul<<<gridSize, blockSize>>>(a,b,c,n);
    }
    else if (type == SHARED) {
        dim3 blockSize(TILE_SIZE,TILE_SIZE);
        dim3 gridSize(n / TILE_SIZE, n / TILE_SIZE);
        sharedMatmul<<<gridSize, blockSize>>>(a,b,c,n);
    }
    // Unknown type
    else
        assert(false);
}



