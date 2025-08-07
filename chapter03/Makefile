.PHONY: all clean 

CC = gcc
NVCC = nvcc

CFLAGS = -I/usr/local/cuda/include -Wall -Wextra -O2 -fPIC
NVCCFLAGS = -O2
LDFLAGS = -lm -L.

PNG_HELPER_DIR = ./png_helper
GRAYSCALE_DIR = ./grayscale
BLUR_DIR = ./blur

all: grayscale.out blur.out

libpng_helper.a: png_helper.o
	ar rcs libpng_helper.a png_helper.o

png_helper.o:
	$(CC) $(CFLAGS) -I$(PNG_HELPER_DIR) -c $(PNG_HELPER_DIR)/png_helper.c -o png_helper.o

grayscale.o:
	$(NVCC) $(NVCCFLAGS) -I$(GRAYSCALE_DIR) -c $(GRAYSCALE_DIR)/grayscale.cu -o grayscale.o

grayscale.out: libpng_helper.a grayscale.o
	$(NVCC) $(NVCCFLAGS) -o grayscale.out grayscale.o libpng_helper.a $(LDFLAGS)

blur.o:
	$(NVCC) $(NVCCFLAGS) -I$(BLUR_DIR) -c $(BLUR_DIR)/blur.cu -o blur.o

blur.out: libpng_helper.a blur.o
	$(NVCC) $(NVCCFLAGS) -o blur.out blur.o libpng_helper.a $(LDFLAGS)

clean:
	rm -f *.o *.a *.out
