// 실험 결과
// 단순 합으로 실험해보니 GPU 메모리 복사 시간 때문에 더 오래 걸렸다.

#include "gpuErrchk.cuh"
#include <time.h>

#define N 30000

/*
// CPU 버전
void add(int* a, int* b, int* c) {
	int tid = 0; // 0번째 CPU임으로, 0에서 시작한다.
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1; // 하나의 CPU를 가지고 있으므로, 하나씩만 증가시킨다.
	}
}

int main(void) {
	int a[N], b[N], c[N];
	// CPU에서 배열 'a'와 'b'를 채운다.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	add(a, b, c);

	// 결과를 화면에 출력한다.
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	return 0;
}
*/

/*
// GPU 버전
__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;	// 이 인덱스의 데이터를 처리한다.
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// GPU 메모리를 할당한다.
	gpuErrchk(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU에서 배열 'a'와 'b'를 채운다.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// 배열 'a'와 'b'를 GPU로 복사한다.
	gpuErrchk(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	// 배열 'c'를 GPU에서 다시 CPU로 복한다.
	gpuErrchk(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	// 결과를 출력한다.
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// GPU에 할당된 메모리를 해제한다.
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
*/


void addCPU(int* a, int* b, int* c) {
	int tid = 0; // 0번째 CPU임으로, 0에서 시작한다.
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1; // 하나의 CPU를 가지고 있으므로, 하나씩만 증가시킨다.
	}
}

__global__ void addGPU(int* a, int* b, int* c) {
	int tid = blockIdx.x;	// 이 인덱스의 데이터를 처리한다.
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	clock_t start, end;



	start = clock();
	////////////////////////////////////////
	// CPU
	int CPU_a[N], CPU_b[N], CPU_c[N];
	// CPU에서 배열 'a'와 'b'를 채운다.
	for (int i = 0; i < N; i++) {
		CPU_a[i] = -i;
		CPU_b[i] = i * i;
	}

	addCPU(CPU_a, CPU_b, CPU_c);

	//// 결과를 화면에 출력한다.
	//for (int i = 0; i < N; i++) {
	//	printf("%d + %d = %d\n", CPU_a[i], CPU_b[i], CPU_c[i]);
	//}

	end = clock();
	printf("CPU time : %lf\n", (double)(end - start) / CLOCKS_PER_SEC);




	start = clock();
	////////////////////////////////////////
	// GPU
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// GPU 메모리를 할당한다.
	gpuErrchk(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU에서 배열 'a'와 'b'를 채운다.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// 배열 'a'와 'b'를 GPU로 복사한다.
	gpuErrchk(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	addGPU << <N, 1 >> > (dev_a, dev_b, dev_c);

	// 배열 'c'를 GPU에서 다시 CPU로 복한다.
	gpuErrchk(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	//// 결과를 출력한다.
	//for (int i = 0; i < N; i++) {
	//	printf("%d + %d = %d\n", a[i], b[i], c[i]);
	//}

	// GPU에 할당된 메모리를 해제한다.
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	end = clock();
	printf("GPU time : %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

	return 0;
}