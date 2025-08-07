#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "png_helper.h"

#define THREADS_X 16
#define THREADS_Y 16

#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err),     \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void cvtGrayScaleKernel(unsigned char *src, unsigned char *dst,
                                   int width, int height, int channels) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if ((col < width) && (row < height)) {
    int oned_idx = col + row * width;
    unsigned char r = src[oned_idx * channels + 0];
    unsigned char g = src[oned_idx * channels + 1];
    unsigned char b = src[oned_idx * channels + 2];
    unsigned char luminance =
        (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    dst[oned_idx] = luminance;
  }
}

void cvtGrayScaleDevice(image_t *src, image_t *dst) {
  int width = src->width;
  int height = src->height;

  dst->width = width;
  dst->height = height;
  dst->channels = 1;

  int src_size = sizeof(unsigned char) * width * height * src->channels;
  int dst_size = sizeof(unsigned char) * width * height * dst->channels;

  unsigned char *d_src;
  cudaMalloc(&d_src, src_size);
  cudaCheckError();

  unsigned char *d_dst;
  cudaMalloc(&d_dst, dst_size);
  cudaCheckError();

  cudaMemcpy(d_src, src->data, src_size, cudaMemcpyHostToDevice);
  cudaCheckError();

  dim3 block_dim{THREADS_X, THREADS_Y, 1};
  unsigned int grid_x = ceil((double)width / THREADS_X);
  unsigned int grid_y = ceil((double)height / THREADS_Y);
  dim3 grid_dim{grid_x, grid_y, 1};
  cvtGrayScaleKernel<<<grid_dim, block_dim>>>(d_src, d_dst, width, height,
                                              src->channels);
  cudaCheckError();

  cudaDeviceSynchronize();

  dst->data = (unsigned char *)malloc(dst_size);
  cudaMemcpy(dst->data, d_dst, dst_size, cudaMemcpyDeviceToHost);
  cudaCheckError();

  cudaFree(d_src);
  cudaFree(d_dst);
}

void cvtGrayScaleHost(image_t *src, image_t *dst) {

  dst->height = src->height;
  dst->width = src->width;
  dst->channels = 1;
  dst->data = (unsigned char *)malloc(src->width * src->height);

  for (int i = 0; i < (dst->width * dst->height); ++i) {
    if (src->channels >= 3) {
      unsigned char r = src->data[i * src->channels + 0];
      unsigned char g = src->data[i * src->channels + 1];
      unsigned char b = src->data[i * src->channels + 2];
      unsigned char luminance =
          (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
      dst->data[i] = luminance;
    } else {
      // already grayscale
      dst->data[i] = src->data[i];
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc != 3) {
    const char *usage = "%s [input.png] [output.png]\n";
    fprintf(stderr, usage, argv[0]);
    exit(1);
  }

  char *input_file = argv[1];
  char *output_file = argv[2];

  image_t image;
  read_image(input_file, &image);

  image_t gray;
  gray.data = NULL;

  // cvtGrayScaleHost(&image, &gray);
  cvtGrayScaleDevice(&image, &gray);

  write_image(output_file, &gray);

  free_image(&image);

  return 0;
}