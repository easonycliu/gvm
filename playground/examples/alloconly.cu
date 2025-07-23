#include <cuda_runtime.h>
#include <iostream>

int main() {
	void *dummy;
	std::size_t size = 2097152L * 256;
	cudaMalloc(&dummy, size);
	std::cout << "Memory allocated" << std::endl;
	std::cin.get();
	cudaMemset(dummy, 1, size);
	std::cout << "Memory touched" << std::endl;
	std::cin.get();
	// cudaFree(dummy);
	std::cout << "Memory freed" << std::endl;
	std::cin.get();
	return 0;
}
