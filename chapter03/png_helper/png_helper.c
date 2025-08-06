#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "png_helper.h"

void read_image(const char *file_name, image_t *img) {
  int width, height, channels;

  unsigned char *data = stbi_load(file_name, &width, &height, &channels, 0);
  if (data == NULL) {
    fprintf(stderr, "Error reading image %s\n", file_name);
    exit(1);
  }

  img->width = width;
  img->height = height;
  img->channels = channels;
  img->data = data;

  printf("Read image %s (%d, %d) with %d channels\n", file_name, width, height, channels);
}

void write_image(const char *file_name, image_t *img) {
  int rtn = stbi_write_png(file_name, img->width, img->height, img->channels,
                           img->data, img->width * img->channels);
  if (rtn == 0) {
    fprintf(stderr, "Error writing image %s\n", file_name);
  } else {
    printf("Image saved as %s\n", file_name);
  }
}

void free_image(image_t *img) { stbi_image_free(img->data); }
