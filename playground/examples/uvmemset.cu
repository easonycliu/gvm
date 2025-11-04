#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <size_in_bytes>\n", argv[0]);
        return 1;
    }

    size_t size = strtoull(argv[1], nullptr, 10);
    if (size == 0) {
        fprintf(stderr, "Invalid size\n");
        return 1;
    }

    // Allocate UVM memory
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        return 1;
    }


    unsigned char val = 0;
    while (true) {
        // memset with the current val
        auto start = std::chrono::steady_clock::now();
        err = cudaMemset(ptr, val, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
            break;
        }

        // Synchronize to ensure memset completes before next iteration
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
            break;
        }
        auto end = std::chrono::steady_clock::now();

        val = (val + 1) % 256;

        // Print once per second
        printf("Iteration takes: %zums\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        fflush(stdout);
    }

    cudaFree(ptr);
    return 0;
}
