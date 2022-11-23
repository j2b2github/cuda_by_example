// ���� ���
// �ܼ� ������ �����غ��� GPU �޸� ���� �ð� ������ �� ���� �ɷȴ�.

#include "gpuErrchk.cuh"
#include <time.h>

#define N 30000

/*
// CPU ����
void add(int* a, int* b, int* c) {
	int tid = 0; // 0��° CPU������, 0���� �����Ѵ�.
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1; // �ϳ��� CPU�� ������ �����Ƿ�, �ϳ����� ������Ų��.
	}
}

int main(void) {
	int a[N], b[N], c[N];
	// CPU���� �迭 'a'�� 'b'�� ä���.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	add(a, b, c);

	// ����� ȭ�鿡 ����Ѵ�.
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	return 0;
}
*/

/*
// GPU ����
__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;	// �� �ε����� �����͸� ó���Ѵ�.
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	// GPU �޸𸮸� �Ҵ��Ѵ�.
	gpuErrchk(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU���� �迭 'a'�� 'b'�� ä���.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// �迭 'a'�� 'b'�� GPU�� �����Ѵ�.
	gpuErrchk(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add << <N, 1 >> > (dev_a, dev_b, dev_c);

	// �迭 'c'�� GPU���� �ٽ� CPU�� ���Ѵ�.
	gpuErrchk(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	// ����� ����Ѵ�.
	for (int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// GPU�� �Ҵ�� �޸𸮸� �����Ѵ�.
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
*/


void addCPU(int* a, int* b, int* c) {
	int tid = 0; // 0��° CPU������, 0���� �����Ѵ�.
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1; // �ϳ��� CPU�� ������ �����Ƿ�, �ϳ����� ������Ų��.
	}
}

__global__ void addGPU(int* a, int* b, int* c) {
	int tid = blockIdx.x;	// �� �ε����� �����͸� ó���Ѵ�.
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main(void) {
	clock_t start, end;



	start = clock();
	////////////////////////////////////////
	// CPU
	int CPU_a[N], CPU_b[N], CPU_c[N];
	// CPU���� �迭 'a'�� 'b'�� ä���.
	for (int i = 0; i < N; i++) {
		CPU_a[i] = -i;
		CPU_b[i] = i * i;
	}

	addCPU(CPU_a, CPU_b, CPU_c);

	//// ����� ȭ�鿡 ����Ѵ�.
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

	// GPU �޸𸮸� �Ҵ��Ѵ�.
	gpuErrchk(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	// CPU���� �迭 'a'�� 'b'�� ä���.
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// �迭 'a'�� 'b'�� GPU�� �����Ѵ�.
	gpuErrchk(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	addGPU << <N, 1 >> > (dev_a, dev_b, dev_c);

	// �迭 'c'�� GPU���� �ٽ� CPU�� ���Ѵ�.
	gpuErrchk(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	//// ����� ����Ѵ�.
	//for (int i = 0; i < N; i++) {
	//	printf("%d + %d = %d\n", a[i], b[i], c[i]);
	//}

	// GPU�� �Ҵ�� �޸𸮸� �����Ѵ�.
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	end = clock();
	printf("GPU time : %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

	return 0;
}