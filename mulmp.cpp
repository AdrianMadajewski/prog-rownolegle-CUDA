#include <omp.h>
#include <random>
#include <ctime>
#include <iostream>
#include <cstring>
#include <iomanip>

constexpr int N = (1 << 10);
constexpr int RAND_LOW = 1;
constexpr int RAND_HIGH = 10;
constexpr float EPSILON = 0.001;

inline float getRandomValue(const int low, const int high)
{
    return low + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (high - low)));
}

void randomizeMatrices(float* A, float* B, float *C, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            A[i * size + j] = getRandomValue(RAND_LOW, RAND_HIGH);
            B[i * size + j] = getRandomValue(RAND_LOW, RAND_HIGH);
            C[i * size + j] = 0.0f;
        }
    }
}

void printMatrix(const float* matrix, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
            std::cout << matrix[i * size + j] << ' ';
        std::cout << '\n';
    }
       
}

void multiplyMatrixIKJ(
    const float* A,
    const float* B, 
    float* C, 
    const int size,
    const int threads)
{
    // #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
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

int main()
{
    srand(time(nullptr));

    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    randomizeMatrices(A, B, C, N);

    double start = omp_get_wtime();
    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    multiplyMatrixIKJ(A, B, C, N, threads);
    double end = omp_get_wtime();
    double duration = end - start;
    std::cout << "[CPU -fopenmp] Finished in: " << duration * 1000 << " [ms] on " << threads << " threads\n";

    // bool correct = verify(A, B, C, N, EPSILON);
    // std::cout << (correct ? "COMPLETED WITH SUCCESS" : "COMPLETED WITH ERROR") << '\n';

    // printMatrix(C, N);

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}