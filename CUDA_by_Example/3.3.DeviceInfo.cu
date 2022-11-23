#include "gpuErrchk.cuh"

int main() {
	cudaDeviceProp prop;

	int count;
	gpuErrchk(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		gpuErrchk(cudaGetDeviceProperties(&prop, i));

		printf(" --- Getneral Infomation for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execition timeout :");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("\n --- Memory Informaion for device %d ---\n", i);
		printf("Total global mem: %ld\n", (long)prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", (long)prop.totalConstMem);
		printf("Max mem pitch: %ld\n", (long)prop.memPitch);
		printf("Texture Alignment: %ld\n", (long)prop.textureAlignment);

		printf("\n --- MP Informaion for device %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", (long)prop.sharedMemPerBlock);
		printf("Registers per ml: %d\n", prop.regsPerBlock);
		printf("Thread in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dismensions: (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1],
			prop.maxThreadsDim[2]);
		printf("Max gris dimensions: (%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1],
			prop.maxGridSize[2]);
		printf("\n");
	}

	return 0;
}
