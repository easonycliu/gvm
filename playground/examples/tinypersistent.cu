#include <cuda_runtime.h>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

__global__ void touch_kernel(volatile char *dummy, std::size_t bytes) {
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

	std::size_t start_index = (bytes / total_threads) * global_id;
	std::size_t end_index = min((bytes / total_threads) * (global_id + 1), bytes);
	for (std::size_t i = start_index; i < end_index; ++i) {
        (reinterpret_cast<char *>(reinterpret_cast<std::uintptr_t>(dummy) + i))[0] = 1;
    }
}

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "Usage: ./uvmalloc <bytes>" << std::endl;
		return 1;
	}

    int device = 0;
	cudaGetDevice(&device);

    std::size_t bytes = std::stoull(argv[1]);
    if (bytes <= 0) {
        std::cerr << "Bytes must be a positive value" << std::endl;
        return 1;
    }
    char *dummy;

	std::cerr << "To allocate " << bytes << " bytes" << std::endl;
    cudaMallocManaged((void **)&dummy, bytes);

    std::cerr << "Allocated " << bytes << " bytes" << std::endl;

	volatile bool stop_flag = false;
	std::thread t = std::thread([&stop_flag, dummy, bytes] () {
		std::cerr << "My thread id is " << gettid() << std::endl;
		while (!stop_flag) {
			touch_kernel<<<16, 64>>>(dummy, bytes);
			cudaDeviceSynchronize();
		}
		std::cerr << "Standby thread exit" << std::endl;
	});

    std::cerr << "Press enter to exit" << std::endl;

	std::cin.get();

	stop_flag = true;
	t.join();

	std::cerr << "Touched " << bytes << " bytes, press to swapout" << std::endl;
	std::cin.get();

	std::chrono::high_resolution_clock::time_point start_swapout = std::chrono::high_resolution_clock::now();
	cudaMemPrefetchAsync(dummy, bytes, cudaCpuDeviceId, 0);
	std::chrono::high_resolution_clock::time_point end_swapout = std::chrono::high_resolution_clock::now();
	std::cerr << "Swapped out in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_swapout - start_swapout).count() << " ms" << std::endl;

	std::cerr << "Swapping out async" << std::endl;

    cudaDeviceSynchronize();
    cudaFree(dummy);

    return 0;
}
