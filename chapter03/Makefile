.PHONY: all clean 

CC = gcc
NVCC = nvcc

CFLAGS = -Wall -Wextra -O2 -fPIC
NVCCFLAGS = -O2
LDFLAGS = -lm

PNG_HELPER_DIR = ./png_helper
GRAYSCALE_DIR = ./grayscale

libpng_helper.a: png_helper.o
	ar rcs libpng_helper.a png_helper.o

png_helper.o:
	$(CC) $(CFLAGS) -I$(PNG_HELPER_DIR) -c $(PNG_HELPER_DIR)/png_helper.c -o png_helper.o

grayscale.o:
	$(CC) $(CFLAGS) -I$(GRAYSCALE_DIR) -c $(GRAYSCALE_DIR)/main.c -o grayscale.o

grayscale.out: libpng_helper.a grayscale.o
	$(CC) $(CFLAGS) -o grayscale.out grayscale.o libpng_helper.a $(LDFLAGS)

clean:
	rm -f *.o *.a *.out
