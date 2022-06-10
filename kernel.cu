#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"
#include "device_functions.h"

#include <cstdio>
#include <random>
#include <ctime>
#include <iostream>
#include <cassert>

// https://stackoverflow.com/questions/35535831/is-there-any-difference-between-cudamallochost-and-cudahostalloc-without-spe
// https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/02_matrix_mul/tiled/mmul.cu

constexpr int N = 1 << 10;					// 1024
constexpr int SHARED_MEMORY_SIZE = 1 << 10; // 1024

constexpr int BYTES = N * N * sizeof(int);
constexpr int RAND_LOW = 0;
constexpr int RAND_HIGH = 1;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int getRandomValue(const int low, const int high);
void randomizeMatrix(int* matrix, const int size);
void printMatrix(int* matrix, const int size);
void verify(int* A, int* B, int* C, const int N);

// Kernel
__global__ void multiplyKernel(const int* __restrict__ A, 
	const int* __restrict__ B, 
	int* __restrict__ C, 
	const int size);

int main(int argc, char **argv)
{
	srand(time(NULL));

	int* A_host = nullptr;
	int* B_host = nullptr;
	int* C_host = nullptr;

	// Allocate memory on host for A, B, C matrices
	// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks

	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	checkCudaErrors(cudaHostAlloc(&A_host, BYTES, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&B_host, BYTES, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&C_host, BYTES, cudaHostAllocMapped));

	// cudaDeviceSynchronize();

	randomizeMatrix(A_host, N);
	randomizeMatrix(B_host, N);

	int* A_device = nullptr;
	int* B_device = nullptr;
	int* C_device = nullptr;

	// Zero copy

	// Allocate memory on device
	checkCudaErrors(cudaMalloc(&A_device, BYTES));
	checkCudaErrors(cudaMalloc(&B_device, BYTES));
	checkCudaErrors(cudaMalloc(&C_device, BYTES));

	// Sync device memory to host memory
	checkCudaErrors(cudaHostGetDevicePointer(&A_device, A_host, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&B_device, B_host, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&C_device, C_host, 0));

	int BLOCK_SIZE = 64;	
	int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	multiplyKernel <<<grid, threads >>> (A_device, B_device, C_device, N);

	cudaDeviceSynchronize();

	verify(A_host, B_host, C_host, N);
	// printMatrix(C_host, N);

	printf("COMPLETED WITH SUCCESS\n");
	return 0;
}

__global__ void multiplyKernel(const int* __restrict__ A, 
	const int* __restrict__ B, 
	int* __restrict__ C, 
	const int size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int shared_A[SHARED_MEMORY_SIZE];
	__shared__ int shared_B[SHARED_MEMORY_SIZE];

	// No alliasing
	int sum = 0;
	for (int i = 0; i < size; i+= blockDim.x)
	{
		// Load elements from tile to shared memory
		int access_tile = threadIdx.y * blockDim.x + threadIdx.x;
		shared_A[access_tile] = A[row * N + i + threadIdx.x];
		shared_B[access_tile] = B[i * N + threadIdx.y * N + col];

		// Intellisense treats it as undefined :(
		__syncthreads();

		// Matrix multiply
		for (int j = 0; j < blockDim.x; j++)
		{
			sum += shared_A[threadIdx.y * blockDim.x + j] * shared_B[j * blockDim.x + threadIdx.x];
		}

		// Intellisense treats it as undefined :(
		__syncthreads();
	}

	// Assign computed result to matrix
	C[row * size + col] = sum;
}

void randomizeMatrix(int* matrix, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			matrix[i * size + j] = 1;
		}
	}
}

void printMatrix(int* matrix, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		std::cout << '\n';
		for (int j = 0; j < size; ++j)
		{
			printf("[%d,%d]=%d\n", i, j, matrix[i * size + j]);
		}
	}
	std::cout << '\n';
}

int getRandomValue(const int low, const int high)
{
	return low + static_cast<int>(rand()) * 
		(static_cast<int>(high - low) / RAND_MAX);
}

void verify(int* A, int* B, int* C, const int N)
{
	// For every row...
	for (int i = 0; i < N; i++) {
		// For every column...
		for (int j = 0; j < N; j++) {
			// For every element in the row-column pair
			int tmp = 0;
			for (int k = 0; k < N; k++) {
				// Accumulate the partial results
				tmp += A[i * N + k] * B[k * N + j];
			}

			// Check against the CPU result
			assert(tmp == C[i * N + j]);
		}
	}
}