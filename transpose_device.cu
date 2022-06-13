#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */


#define TILE_SIZE 32

__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    // reads and writes to and from global memory are slow.
    // Also, the method of reading and writing creates memory
    // bank conflicts. 

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;
    for (; j < end_j; j++)
//non coalesced read/write to/from global memory.
        output[j + n * i] = input[i + n * j];
}


__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    // This is our version of the shared memory transpose.
    // By pulling data into local blocks, we speed up the calculations.
    // Global memory reads and writes are also coalesced.
    // Lastly, there are memory bank conflicts when using a size of 
    // [32][32].  (Actually, its a 32-way bank conflict)
    // However, these are removed entirely by padding the
    // the shared array, so it is of size [32][33].


    const int length = TILE_SIZE/blockDim.y;
    const int numhops = blockDim.y;

// padded shared,local array
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + length* blockDim.y * blockIdx.y;
    int index = j*n+i;

    for(int k =0; k< length; k++){
// coalesced read from global memory, coalesced write to shared memory. 
// no bank conflicts.
	tile[threadIdx.y+(k*numhops)][threadIdx.x] = input[index + k*numhops*n];
    }

    __syncthreads();
  
    i = threadIdx.x+  blockDim.y * blockIdx.y*length;
    j = threadIdx.y + blockDim.x * blockIdx.x; 
    index = j*n+i;

    for (int k =0; k < length; k++){
// coalesced read from shared memory, coalesced write to global memory.
// no bank conflicts.
     	output[index+k*numhops*n] = tile[threadIdx.x][threadIdx.y +(k*numhops)];
    }
}

// */


__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.
    //
    // The compiler already performs some loop unrolling automatically. 
    // Adding unroll directives didn't affect the efficiency much.
    // I couldn't find much to improve on the version above. 

    const int length = TILE_SIZE/blockDim.y;
    const int numhops = blockDim.y;

// padding the matrix removes shared memory bank conflicts.
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + length* blockDim.y * blockIdx.y;
    int index = j*n+i;

    #pragma unroll 
    for(int k =0; k< length; k++){
// coalesced read from global memory, and coalesced write to shared memory
// no memory bank conflicts.
	tile[threadIdx.y+(k*numhops)][threadIdx.x] = input[index + k*numhops*n];
    }

    __syncthreads();
  
    i = threadIdx.x+  blockDim.y * blockIdx.y*length;
    j = threadIdx.y + blockDim.x * blockIdx.x; 
    index = j*n+i;
    #pragma unroll 
    for (int k =0; k < length; k++){
// coalesced read from shared memory, coalesced write to global memory.
// no bank conflicts.
     	output[index+k*numhops*n] = tile[threadIdx.x][threadIdx.y +(k*numhops)];
    }
// */
}




void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{

   if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(TILE_SIZE,TILE_SIZE/4);
        dim3 gridSize(n / TILE_SIZE, n / TILE_SIZE);
	 shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    }
    else if (type == OPTIMAL) {
        dim3 blockSize(TILE_SIZE, TILE_SIZE/4);
        dim3 gridSize(n / TILE_SIZE, n / TILE_SIZE);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);

	

}
