#ifndef PNG_HELPER_STUB_H
#define PNG_HELPER_STUB_H

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