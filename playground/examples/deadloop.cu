#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

__global__ void quick_kernel() {
	unsigned long long start = clock64();
    unsigned long long freq  = 1'410'000'000ULL; // cycles per second

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x == 0 && y == 0)
		printf("x is %d, y is %d\n", x, y);

    while (true) {
		if (x == 0 && y == 0) {
			unsigned long long now = clock64();
        	if (now - start >= freq) {
        	    start = now;
				printf("GPU says: Hello %lld\n", now);
        	}
		}
    }
}

int main() {
	std::cout << "Starting kernel execution...\n";
	quick_kernel<<<128, 256>>>();
	while (true);
    return 0;
}
