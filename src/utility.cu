#include <png.h>
using namespace std;
#include "utility.h"

float randomFloat() {
    return (float)(rand()) / (float)(RAND_MAX);
}

int randomInt(int a, int b)
{
    if (a > b)
        return randomInt(b, a);
    if (a == b)
        return a;
    return a + (rand() % (b - a));
}

float randomFloat(int a, int b)
{
    if (a > b)
        return randomFloat(b, a);
    if (a == b)
        return a;
 
    return (float)randomInt(a, b) + randomFloat();
}


void save_png(unsigned char* image, int width, int height, const char* filename) {
    // Three channels (RGB).
    // Can also use 4 channels (RGBA) for transparency, but then the buffer need to be 4*width*height

    png_bytep row_pointers[height];
    for (int i = 0; i < height; ++i) {
        row_pointers[i] = (png_bytep) (image + i * width * 3);
    }

    FILE* fp = fopen(filename, "wb"); // Create a file for writing (binary mode)
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_set_IHDR(
        png,
        info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);
    fclose(fp);

    png_destroy_write_struct(&png, &info);
    printf("Saved image to %s\n", filename);
}

void image(int N, Electron* electrons, int iteration) {
    int width = 50;
    int height = 50;

    unsigned char *image = (unsigned char*)malloc(width * height * 3);

    // Create a checkboard pattern with 16x16 squares
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[(y * width + x) * 3 + 0] = 255;
            image[(y * width + x) * 3 + 1] = 255;
            image[(y * width + x) * 3 + 2] = 255;
        }
    }

    for (int i = 0; i < N; i++) {
        if (i == 0) {
            image[((int)(50 - electrons[i].position.y) * width + (int)(electrons[i].position.x)) * 3 + 0] = 255;
            image[((int)(50 - electrons[i].position.y) * width + (int)(electrons[i].position.x)) * 3 + 1] = 0;
            image[((int)(50 - electrons[i].position.y) * width + (int)(electrons[i].position.x)) * 3 + 2] = 0;
        }
        else {
            image[((int)(50 - electrons[i].position.y) * width + (int)(electrons[i].position.x)) * 3 + 0] = 0;
            image[((int)(50 - electrons[i].position.y) * width + (int)(electrons[i].position.x)) * 3 + 1] = 0;
            image[((int)(50 - electrons[i].position.y) * width + (int)(electrons[i].position.x)) * 3 + 2] = 0;
        }
    }
    char filename[11];
    sprintf(filename, "test_%0d.png", iteration);
    save_png(image, width, height, filename);

    // Free byte buffer
    free(image);

}

