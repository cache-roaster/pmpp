#include <stdio.h>
#include <stdlib.h>

#include "png_helper.h"

void cvtGrayScaleHost(image_t *src, image_t *dst) {
  if (dst->data != NULL) {
    free(dst->data);
  }

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

  image_t gray_image;
  cvtGrayScaleHost(&image, &gray_image);

  write_image(output_file, &gray_image);

  free_image(&image);

  return 0;
}