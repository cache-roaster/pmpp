#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <random>
#include <vector>

#ifndef N
#define N 8192
#endif

#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err),     \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define THREADS_X 16
#define THREADS_Y 16
#define BLOCKS_X(width) ((width + THREADS_X - 1) / THREADS_X)
#define BLOCKS_Y(height) ((height + THREADS_Y - 1) / THREADS_Y)

#define TILE_WIDTH 32

__global__ void matmul_naive(int n, double *matA, double *matB, double *matC) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if ((col < n) && (row < n)) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum += matA[i + row * n] * matB[col + i * n];
    }
    matC[col + row * n] = sum;
  }
}

__global__ void matmul_tiled(int n, double *matA, double *matB, double *matC) {

  __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  double val = 0;
  for (int ph = 0; ph < n / TILE_WIDTH; ++ph) {
    if ((row < n) && (ph * TILE_WIDTH + tx) < n) {
      Mds[ty][tx] = matA[row * n + ph * TILE_WIDTH + tx];
    }
    if ((ph * TILE_WIDTH + ty) < n && (col < n)) {
      Nds[ty][tx] = matB[row * n + ph * TILE_WIDTH + tx];
    }

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      val += Mds[ty][k] * Nds[k][tx];
    }

    __syncthreads();
  }
  if ((row < n) && (col < n)) {
    matC[row * n + col] = val;
  }
}

std::vector<double> generateMatrix(int n, int lower = 0, int upper = 99,
                                   double scale = 1.0) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 100);

  std::vector<double> arr(n * n);
  std::generate(arr.begin(), arr.end(),
                [&]() { return static_cast<double>(dis(gen)) * scale; });

  return arr;
}

double matmul_stub(int n, std::vector<double> &matA, std::vector<double> &matB,
                   std::vector<double> &matC) {
  double *hA = matA.data();
  double *hB = matB.data();
  double *hC = matC.data();

  int size = sizeof(double) * n * n;
  double *dA;
  double *dB;
  double *dC;
  cudaMalloc((void **)&dA, size);
  cudaMalloc((void **)&dB, size);
  cudaMalloc((void **)&dC, size);

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

  dim3 threads(THREADS_X, THREADS_Y);
  dim3 blocks(BLOCKS_X(n), BLOCKS_Y(n));

  cudaEventRecord(begin);
  // matmul_naive<<<blocks, threads>>>(n, dA, dB, dC);
  matmul_tiled<<<blocks, threads>>>(n, dA, dB, dC);
  cudaEventRecord(end);

  cudaEventSynchronize(end);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, begin, end);

  // printf("Kernel execution time: %f ms\n", milliseconds);

  cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return milliseconds;
}

bool checkCorrectness() {
  std::vector<double> matA{1, 3, 5, 7};
  std::vector<double> matB{2, 4, 6, 8};
  std::vector<double> matC(4);
  matmul_stub(2, matA, matB, matC);

  if ((matC[0] != 20) || (matC[1] != 28) || (matC[2] != 52) ||
      (matC[3] != 76)) {
    return false;
  }
  return true;
}

double benchmark(int n) {
  std::vector<double> matA = generateMatrix(n);
  std::vector<double> matB = generateMatrix(n);
  std::vector<double> matC(n * n);

  double time_ms = matmul_stub(n, matA, matB, matC);
  double flops = 2.0 * n * n * n;
  return (double)flops / time_ms / 1000.0 / 1000.0 / 1000.0;
}

int main(int argc, char *argv[]) {
  int dev_cnt;
  cudaGetDeviceCount(&dev_cnt);
  if (dev_cnt < 1) {
    std::cerr << "No CUDA device found!" << std::endl;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "GPU 0: " << prop.name << std::endl;

  double gflops = benchmark(N);
  std::cout << "GFlop/s: " << gflops << std::endl;

  return 0;
}
