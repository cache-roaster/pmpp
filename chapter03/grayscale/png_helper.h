#ifndef PNG_HELPER_STUB_H
#define PNG_HELPER_STUB_H

typedef struct {
  unsigned char *data;
  int width;
  int height;
  int channels;
} image_t;

extern "C" {
int read_image(const char *filename, image_t *img);
int write_image(const char *filename, image_t *img);
void free_image(image_t *img);
}

#endif