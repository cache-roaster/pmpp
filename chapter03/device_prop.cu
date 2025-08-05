#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int dev_cnt;
  cudaGetDeviceCount(&dev_cnt);
  printf("CUDA device count: %d\n", dev_cnt);

  cudaDeviceProp dev_prop;
  for (int i = 0; i < dev_cnt; ++i) {
    cudaGetDeviceProperties(&dev_prop, i);
    printf("Get device property for GPU %d:\n", i);
    printf("Name: %s\n", dev_prop.name);
    printf("Clock rate: %d\n", dev_prop.clockRate);
    printf("Max Threads Per Block: %d\n", dev_prop.maxThreadsPerBlock);
    printf("SM Count: %d\n", dev_prop.multiProcessorCount);
    printf("Max Threads Dim(x, y, z): %d, %d, %d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    printf("Max Grid Size(x, y, z): %d, %d, %d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
  }

  return 0;
}