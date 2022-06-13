
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "matmul_device.cuh"
#include "ta_utilities.hpp"
/*
 * NOTE: You can use this macro to easily check cuda error codes
 * and get more information.
 * 
 * Modified from:
 * http://stackoverflow.com/questions/14038589/
 *         what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}


// a is gold; b is to be checked 
void checkMatmul(float *a, float *b, int n) {
    bool correct = true;
    double anorm = 0;
    double dnorm = 0; // norm of a-b
    for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	    anorm += a[i*n+j]*a[i*n+j];
	    dnorm += (a[i*n+j]-b[i*n+j])*(a[i*n+j]-b[i*n+j]);
	}
    }
    anorm = sqrt(anorm);
    dnorm = sqrt(dnorm);
    // lazy check:
    if (dnorm/anorm > 0.01) {
	correct = false;
	printf("Incorrect result! Aborting...\n");
    }

    assert(correct);
}

// scalar CPU matmul using ikj loop. 
void cpuMatmul(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
		c[i*n+j] += a[i*n+k]*b[k*n+j];
	    }
	}
    }
}

/*
 * Fills fill with random numbers is [0, 1]. Size is number of elements to
 * assign.
 */
void randomFill(float *fill, int size) {
    for (int i = 0; i < size; i++) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        fill[i] = r;
    }
}

int main(int argc, char *argv[]) {

    
    // Seed random number generator
    srand(2016);

    std::string kernel = "all";
    int size_to_run = -1;

    // Check arguments
    assert(argc <= 2);
    if (argc >= 2)
        size_to_run = atoi(argv[1]);

    if (!(size_to_run == -1  ||
         size_to_run == 512  ||
         size_to_run == 1024 ||
         size_to_run == 2048 ||
         size_to_run == 4096))
    {
        fprintf(stderr,
            "Program only designed to run sizes 512, 1024, 2048, 4096\n");
    }

    assert(kernel == "all"     ||
        kernel == "cpu"        ||
//        kernel == "gpu_memcpy" ||
        kernel == "naive"      ||
        kernel == "cache"      ||
        kernel == "shared");

    // Run the transpose implementations for all desired sizes (2^9 = 512, 
    // 2^12 = 4096)
    for (int _i = 9; _i < 13; _i++) {
        int n = 1 << _i;
        if (size_to_run != -1 && size_to_run != n)
            continue;

        assert(n % 64 == 0);

        cudaEvent_t start;
        cudaEvent_t stop;

#define START_TIMER() {                                                        \
            gpuErrChk(cudaEventCreate(&start));                                \
            gpuErrChk(cudaEventCreate(&stop));                                 \
            gpuErrChk(cudaEventRecord(start));                                 \
        }

#define STOP_RECORD_TIMER(name) {                                              \
            gpuErrChk(cudaEventRecord(stop));                                  \
            gpuErrChk(cudaEventSynchronize(stop));                             \
            gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
            gpuErrChk(cudaEventDestroy(start));                                \
            gpuErrChk(cudaEventDestroy(stop));                                 \
        }

        // Initialize timers
        float cpu_ms = -1;
        float gpu_memcpy = -1;
        float naive_gpu_ms = -1;
        float shmem_gpu_ms = -1;
        float optimal_gpu_ms = -1;

        // Allocate host memory
        float *h_a = new float[n*n];
	float *h_b = new float[n*n];
	float *h_c = new float[n*n];
	float *h_cc = new float[n*n];
	float *h_cgold = new float[n*n];

        // Allocate device memory
        float *d_a, *d_b, *d_c;
        gpuErrChk(cudaMalloc(&d_a, n * n * sizeof(float)));
        gpuErrChk(cudaMalloc(&d_b, n * n * sizeof(float)));
	gpuErrChk(cudaMalloc(&d_c, n * n * sizeof(float)));

        // Initialize input data to random numbers in [0, 1]
        randomFill(h_a, n * n);
	randomFill(h_b, n * n);
	randomFill(h_c, n * n);


        // Copy input to GPU
        gpuErrChk(cudaMemcpy(d_a, h_a, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_b, h_b, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_c, h_c, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));


        // CPU implementation
        //if (kernel == "cpu" || kernel == "all") {
	START_TIMER();
	// cpuTranspose(input, output, n);
	cpuMatmul(h_a, h_b, h_cgold, n);
	STOP_RECORD_TIMER(cpu_ms);

	//checkMatmul(d_a, d_b, d_c, n);
	//memset(output, 0, n * n * sizeof(float));

	printf("Size %d naive CPU: %f ms\n", n, cpu_ms);
        //}


        // Naive GPU implementation
        //if (kernel == "naive" || kernel == "all") {
	START_TIMER();
	//cudaTranspose(d_input, d_output, n, NAIVE);
	cudaMatmul(d_a, d_b, d_c, n, NAIVE);
	STOP_RECORD_TIMER(naive_gpu_ms);

	gpuErrChk(cudaMemcpy(h_cc, d_c, n * n * sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkMatmul(h_cgold,h_cc, n);

	//memset(output, 0, n * n * sizeof(float));
	//gpuErrChk(cudaMemset(d_output, 0, n * n * sizeof(float)));

	printf("Size %d naive GPU: %f ms\n", n, naive_gpu_ms);
        //}
        // cache GPU implementation
        //if (kernel == "shmem" || kernel == "all") {
        // Copy input to GPU
        gpuErrChk(cudaMemcpy(d_a, h_a, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_b, h_b, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));
	gpuErrChk(cudaMemcpy(d_c, h_c, n * n * sizeof(float), 
            cudaMemcpyHostToDevice));
	// set L1/shared memory config to 48K/16K
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	START_TIMER();
	cudaMatmul(d_a, d_b, d_c, n,CACHE);
	STOP_RECORD_TIMER(shmem_gpu_ms);

	gpuErrChk(cudaMemcpy(h_cc, d_c, n * n * sizeof(float), 
				cudaMemcpyDeviceToHost));
	checkMatmul(h_cgold,h_cc, n);


	printf("Size %d cache GPU: %f ms\n", n, shmem_gpu_ms);
        //}
	gpuErrChk(cudaMemcpy(d_a, h_a, n * n * sizeof(float),
            cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(d_b, h_b, n * n * sizeof(float),
            cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(d_c, h_c, n * n * sizeof(float),
            cudaMemcpyHostToDevice));
	// set L1/shared memory config to 16K/48K
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        START_TIMER();
        cudaMatmul(d_a, d_b, d_c, n,SHARED);
        STOP_RECORD_TIMER(shmem_gpu_ms);

        gpuErrChk(cudaMemcpy(h_cc, d_c, n * n * sizeof(float),
                                cudaMemcpyDeviceToHost));
        checkMatmul(h_cgold,h_cc, n);


        printf("Size %d shared GPU: %f ms\n", n, shmem_gpu_ms);

        // Free host memory
        //delete[] input;
        //delete[] output;

        // Free device memory
        //gpuErrChk(cudaFree(d_input));
        //gpuErrChk(cudaFree(d_output));

        printf("\n");
    }
}
