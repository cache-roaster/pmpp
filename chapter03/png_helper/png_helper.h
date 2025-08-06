#ifndef PNG_HELPER_H
#define PNG_HELPER_H

#include <stdio.h>
#include <stdlib.h>

// nothings/stb image loader and writer is used in this code:
// https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
// https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
#include "stb_image.h"
#include "stb_image_write.h"

typedef struct {
  unsigned char *data;
  int width;
  int height;
  int channels;
} image_t;

void read_image(const char *file_name, image_t *img);

void write_image(const char *file_name, image_t *img);

void free_image(image_t *img);

#endif