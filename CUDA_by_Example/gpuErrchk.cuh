#include <iostream>
#include <cuda_runtime.h>

// 출처: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// //이걸 포함하니 에러나네...
//#include <assert.h>
//#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
//__device__ void cdpAssert(cudaError_t code, const char* file, int line, bool abort = true)
//{
//	if (code != cudaSuccess)
//	{
//		printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
//		if (abort) assert(0);
//	}
//}