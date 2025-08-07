#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "png_helper.h"

#define BLUR_SIZE 4

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

__global__ void blurKernel(unsigned char *src, unsigned char *dst, int width,
                           int height) {

  const int col = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = threadIdx.y + blockIdx.y * blockDim.y;

  if ((col < width) && row < (height)) {
    int pixels = 0;
    int sum = 0;
    for (int i = (row - BLUR_SIZE + 1); i < (row + BLUR_SIZE); ++i) {
      for (int j = (col - BLUR_SIZE + 1); j < (col + BLUR_SIZE); ++j) {
        if ((i > -1) && (i < height) && (j > -1) && (j < width)) {
          pixels++;
          sum += src[i * width + j];
        }
      }
    }

    dst[row * width + col] = (unsigned char)(sum / pixels);
  }
}

void blurDevice(image_t *src, image_t *dst) {

  const int width = src->width;
  const int height = src->height;

  dst->width = src->width;
  dst->height = src->height;
  dst->channels = 1;

  const int size = sizeof(unsigned char) * width * height;

  unsigned char *d_src;
  cudaMalloc(&d_src, size);
  cudaCheckError();

  unsigned char *d_dst;
  cudaMalloc(&d_dst, size);
  cudaCheckError();

  cudaMemcpy(d_src, src->data, size, cudaMemcpyHostToDevice);
  cudaCheckError();

  dim3 block_dim{THREADS_X, THREADS_Y, 1};
  unsigned int grid_x = ceil((double)width / THREADS_X);
  unsigned int grid_y = ceil((double)height / THREADS_Y);
  dim3 grid_dim{grid_x, grid_y, 1};

  blurKernel<<<block_dim, grid_dim>>>(d_src, d_dst, width, height);
  cudaCheckError();

  cudaDeviceSynchronize();

  dst->data = (unsigned char *)malloc(size);
  cudaMemcpy(dst->data, d_dst, size, cudaMemcpyDeviceToHost);
  cudaCheckError();

  cudaFree(d_src);
  cudaFree(d_dst);
}

void blurHost(image_t *src, image_t *dst) {

  dst->width = src->width;
  dst->height = src->height;
  dst->channels = 1;
  dst->data = (unsigned char *)malloc(src->width * src->height);

  for (int r = 0; r < dst->height; ++r) {
    for (int c = 0; c < dst->width; ++c) {
      int pixels = 0;
      int sum = 0;
      for (int i = (r - BLUR_SIZE + 1); i < (r + BLUR_SIZE); ++i) {
        for (int j = (c - BLUR_SIZE + 1); j < (c + BLUR_SIZE); ++j) {
          if ((i > -1) && (i < dst->height) && (j > -1) && (j < dst->width)) {
            pixels++;
            sum += src->data[i * src->width + j];
          }
        }
      }

      dst->data[r * src->width + c] = (unsigned char)(sum / pixels);
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

  if (image.channels != 1) {
    fprintf(stderr, "Input image has to be black & white!\n");
    exit(1);
  }

  image_t blurred;
  // blurHost(&image, &blurred);
  blurDevice(&image, &blurred);

  write_image(output_file, &blurred);
  free_image(&image);
  free_image(&blurred);

  return 0;
}