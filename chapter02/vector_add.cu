#include "cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void vecAddHost(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAddDevice(float *hA, float *hB, float *hC, int n) {
  const int size = n * sizeof(float);
  float *dA;
  float *dB;
  float *dC;
  cudaMalloc((void **)&dA, size);
  cudaMalloc((void **)&dB, size);
  cudaMalloc((void **)&dC, size);

  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

  vecAddKernel<<<ceil(n / 256.0), 256>>>(dA, dB, dC, n);

  cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

int main() {
  int n = 100000000; // 100M
  float *A = (float *)malloc(n * sizeof(float));
  float *B = (float *)malloc(n * sizeof(float));
  float *C = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < n; ++i) {
    A[i] = i;
    B[i] = n - i;
  }

   vecAddDevice(A, B, C, n);
   // vecAddHost(A, B, C, n);


  printf("%.2f\n", C[0]);

  return 0;
}
