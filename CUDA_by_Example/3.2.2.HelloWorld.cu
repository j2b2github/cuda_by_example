#include <cuda_runtime.h>
#include <iostream>

__global__ void Kernel(void) {

}

int main() {
	Kernel << <1, 1 >> > ();
	printf("Hello, World!\n");
	return 0;
}