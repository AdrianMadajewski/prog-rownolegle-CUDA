#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"

#include <cstdio>
#include <random>
#include <ctime>
#include <iostream>

constexpr int N = 16;
constexpr int SIZE = N * N;
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

// Kernel
__global__ void multiplyKernel(int* A, int* B, int* C, const int size);

int main()
{
	srand(time(NULL));

	int* A_host = nullptr;
	int* B_host = nullptr;
	int* C_host = nullptr;

	// Allocate memory on host for A, B, C matrices
	checkCudaErrors(cudaMallocHost(&A_host, SIZE));
	checkCudaErrors(cudaMallocHost(&B_host, SIZE));
	checkCudaErrors(cudaMallocHost(&C_host, SIZE));

	randomizeMatrix(A_host, N);
	randomizeMatrix(B_host, N);

	// Zero C-result array
	memset(C_host, 0, SIZE);

	//printMatrix(A_host, N);
	//printMatrix(B_host, N);
	//printMatrix(C_host, N);

	int* A_device = nullptr;
	int* B_device = nullptr;
	int* C_device = nullptr;

	// Zero copy

	// Allocate memory on device
	checkCudaErrors(cudaMalloc(&A_device, SIZE));
	checkCudaErrors(cudaMalloc(&B_device, SIZE));
	checkCudaErrors(cudaMalloc(&C_device, SIZE));

	// Sync device memory to host memory
	checkCudaErrors(cudaHostGetDevicePointer(&A_device, A_host, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&B_device, B_host, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&C_device, C_host, 0));

	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	multiplyKernel <<<blocksPerGrid, threadsPerBlock >>> (A_device, B_device, C_device, N);

	cudaDeviceSynchronize();
	printMatrix(C_host, N);

	return 0;
}

__global__ void multiplyKernel(int* A, int* B, int* C, const int size)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int sum = 0;

	__shared__ float Msub[16 * 16];

	if (row < size && col < size)
	{
		for (int i = 0; i < size; ++i)
		{
			printf("A: row=%d,col=%d,i=%d, value=%d\n", row, col, i, A[row * size + i]);
			printf("B: row=%d,col=%d,i=%d, value=%d\n", row, col, i, B[row * size + i]);
			sum += A[row * size + i] * B[i * size + col];
		}
	}
	
	C[row * size + col] = sum;
	// printf("C: row=%d,col=%d, value=%d\n", row, col, sum);
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
	return low + static_cast<int>(rand()) * (static_cast<int>(high - low) / RAND_MAX);
}