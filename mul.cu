#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"
#include "device_functions.h"

#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <iomanip>

__constant__ constexpr int N = 2048; // * 2 = 2048 * 3 = 3072
__constant__ constexpr int BYTES = N * N * sizeof(int);
__constant__ constexpr int THREADS = 32;		 // 32 MAX	-- Warp size: 32
__constant__ constexpr int SUB_SIZE = 32;

constexpr float EPSILON = 0.001f; // Decimal places rounding for error checking
constexpr int RAND_LOW = 1;
constexpr int RAND_HIGH = 10;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline int getRandomValue(const int low, const int high)
{
	return low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (high - low)));
}

void randomizeMatrix(float* matrix, const int size)
{
	for (int i = 0; i < size; ++i)
	{
		for (int j = 0; j < size; ++j)
		{
			matrix[i * size + j] = getRandomValue(RAND_LOW, RAND_HIGH);
		}
	}
}
void printMatrix(const float* matrix, const int size)
{
	for (int row = 0; row < size; ++row)
	{
		std::cout << '\n';
		for (int col = 0; col < size; ++col)
		{
			std::cout << "[row=" << row << ",col=" << col << "]=" << matrix[row * size + col] << '\n';
		}
	}
	std::cout << '\n';
}

bool verify(const float* A, const float* B, const float* C, const int N, const float eps)
{
	for (int row = 0; row < N; ++row) {
		for (int col = 0; col < N; ++col) {
			float tmp = 0;
			for (int index = 0; index < N; ++index) {
				tmp += A[row * N + index] * B[index * N + col];
			}

			// Check against the CPU result
			if (fabs(tmp - C[row * N + col]) >= eps)
			{
				std::cout << std::setprecision(16) << "Expected: " << tmp << ", got: " << C[row * N + col] << '\n';
				return false;
			}

		}
	}
	return true;
}

__device__ inline int index(const int row, const int column, const int width)
{
	return row * width + column;
}

__global__ void multiplyKernel(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	const int size)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	float sum = 0.0f;
	for (int i = 0; i < size; i++)
	{
		sum += A[index(row, i, size)] * B[index(i, col, size)];
	}
	C[index(row, col, size)] = sum;
}

int main(int argc, char** argv)
{
	srand(time(nullptr));

	float* A_host, * B_host, * C_host;
	float* A_device, * B_device, * C_device;

	// Allocate memory on host for A, B, C matrices

	// Prepare paged memory
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	checkCudaErrors(cudaHostAlloc(&A_host, BYTES, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&B_host, BYTES, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&C_host, BYTES, cudaHostAllocMapped));

	randomizeMatrix(A_host, N);
	randomizeMatrix(B_host, N);

	// printMatrix(A_host, N);
	// printMatrix(B_host, N);

	// Set C-matrix to 0s
	checkCudaErrors(cudaMemset(C_host, 0, BYTES));

	// printMatrix(A_host, N);
	// printMatrix(B_host, N);

	// Zero copy

	// Sync device memory to host memory
	checkCudaErrors(cudaHostGetDevicePointer((void**)&A_device, (void*)A_host, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&B_device, (void*)B_host, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&C_device, (void*)C_host, 0));

	int BLOCKS = (int)ceil(N / THREADS); // ceil is unecessary as N divides threads evenly

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	//cudaEvent_t start, stop;
	//checkCudaErrors(cudaEventCreate(&start));
	//checkCudaErrors(cudaEventCreate(&stop));

	//std::cout << "MATRIX-SIZE = " << N << '\n';
	//std::cout << "[DEVICE]: Multiply kernel started\n";

	//checkCudaErrors(cudaEventRecord(start, 0));
	multiplyKernel << <blocks, threads >> > (A_device, B_device, C_device, N);
	//checkCudaErrors(cudaEventRecord(stop, 0));
	//checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaDeviceSynchronize());

	//float msecTotal = 0.0f;
	//checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	//checkCudaErrors(cudaEventDestroy(start));
	//checkCudaErrors(cudaEventDestroy(stop));

	//std::cout << "[DEVICE]: Multiply kernel finished in: " << msecTotal << " [ms]\n";

	// CORRECTION CODE GOES HERE

	//std::cout << "[CPU]: Verify CPU started\n";
	//auto start_cpu = std::chrono::steady_clock::now();
	//auto correct = verify(A_host, B_host, C_host, N, EPSILON);
	//auto stop_cpu = std::chrono::steady_clock::now();
	//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu).count();
	//std::cout << "[CPU]: Verify CPU finished in: " << duration << " [ms]\n";

	//std::cout << (correct ? "COMPLETED SUCCESFULLY\n" : "COMPLETED WITH ERROR (verify)\n");
	// printMatrix(C_host, N);

	// Free host memory
	cudaFreeHost(A_host);
	cudaFreeHost(B_host);
	cudaFreeHost(C_host);

	// No need to free device pointers

	// Restore device
	checkCudaErrors(cudaDeviceReset());
	return 0;
}