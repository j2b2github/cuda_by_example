#include "gpuErrchk.cuh"

int main(void) {
	cudaDeviceProp prop;
	int dev;

	gpuErrchk(cudaGetDevice(&dev));
	printf("ID of current CUDA device: %d\n", dev);

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	gpuErrchk(cudaChooseDevice(&dev, &prop));
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
	gpuErrchk(cudaSetDevice(dev));
}