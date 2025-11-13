// simple_kernels.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                    __FILE__, __LINE__, cudaGetErrorString(err__));              \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// -------------------------
// Kernels
// -------------------------

// out: [rows * cols] row-major
extern "C" __global__
void zeros_kernel(float* out, int rows, int cols) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        out[idx] = 0.0f;
    }
}

// out: [rows * cols] row-major
extern "C" __global__
void ones_kernel(float* out, int rows, int cols) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        out[idx] = 1.0f;
    }
}

// out: [n * n] identity matrix, row-major
extern "C" __global__
void eye_kernel(float* out, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        out[row * n + col] = (row == col) ? 1.0f : 0.0f;
    }
}

// C = A (M×K) * B (K×N), row-major
extern "C" __global__
void matmul_kernel(const float* A,
                   const float* B,
                   float* C,
                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..N)

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -------------------------
// Simple test / demo
// -------------------------

static void print_matrix(const char* name, const float* h, int rows, int cols) {
    printf("%s =\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.2f ", h[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    const int n = 4;          // size for eye and matmul
    const int rows = n, cols = n;
    const int size = rows * cols;
    const size_t bytes = size * sizeof(float);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    dim3 block1D(256);
    dim3 grid1D((size + block1D.x - 1) / block1D.x);

    dim3 block2D(16, 16);
    dim3 grid2D((cols + block2D.x - 1) / block2D.x,
                (rows + block2D.y - 1) / block2D.y);

    // 1) A = eye, B = ones, C = A*B
    eye_kernel<<<grid2D, block2D>>>(dA, n);
    ones_kernel<<<grid1D, block1D>>>(dB, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    matmul_kernel<<<grid2D, block2D>>>(dA, dB, dC, rows, cols, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float hA[size], hB[size], hC[size];
    CHECK_CUDA(cudaMemcpy(hA, dA, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hB, dB, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    print_matrix("A (eye)",  hA, rows, cols);
    print_matrix("B (ones)", hB, rows, cols);
    print_matrix("C = A*B",  hC, rows, cols);

    // 2) Test zeros_kernel: C = 0
    zeros_kernel<<<grid1D, block1D>>>(dC, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    print_matrix("C after zeros_kernel", hC, rows, cols);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return 0;
}

