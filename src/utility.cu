#include <png.h>
#include "utility.h"

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

void draw_particle (unsigned char* image, float x, float y, int height, int width) { 
    for (int i = -5; i <= 5; i++) {
        for (int j = -5; j <= 5; j++) { 
            
            if (y+j > height || y+j < 0 || x+i > width || x+i < 0) continue;

            image[((int)(height - (y+j)) * width + (int)(x+i)) * 3 + 0] = 0;
            image[((int)(height - (y+j)) * width + (int)(x+i)) * 3 + 1] = 0;
            image[((int)(height - (y+j)) * width + (int)(x+i)) * 3 + 2] = 0;
        }
    }
}

void image(int N, Electron* electrons, int iteration) {
    int width = 500;
    int height = 500;

    unsigned char *image = (unsigned char*)malloc(width * height * 3 + 1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[(y * width + x) * 3 + 0] = 255;
            image[(y * width + x) * 3 + 1] = 255;
            image[(y * width + x) * 3 + 2] = 255;
        }
    }

    for (int i = 0; i < N; i++) {
        draw_particle (image, electrons[i].position.x, electrons[i].position.y, height, width);
    }
    char filename[40];
    sprintf(filename, "./out/visualization/test_%04d.png", iteration);
    save_png(image, width, height, filename);

    // Free byte buffer
    free(image);

}


// Timing
chrono::time_point<high_resolution_clock> start_cpu_timer(){
    return high_resolution_clock::now();
}

double end_cpu_timer(chrono::time_point<high_resolution_clock> start){
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count() / 1000000.0;
}

void printTimingHeader(FILE* os){
    fprintf(os, "func,init n,iterations,block size,sleep time,collision chance,final n,time\n");
}

void printTimingData(const TimingData& data, FILE* os){
    fprintf(os, "%s,%d,%d,%d,%d,%f,%d,%f\n", data.function.c_str(), data.init_n, data.iterations, data.block_size, data.sleep_time, data.split_chance, data.final_n, data.time);
}

void printCSV(const vector<TimingData>& data, FILE* os){
    printTimingHeader(os);
    for(const auto& d : data){
        printTimingData(d, os);
    }
}

void printCSV(const vector<TimingData>& data, string filename){
    FILE* os = fopen(filename.c_str(), "w");
    printCSV(data, os);
    fclose(os);
}

void checkCudaError(string context) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\nAt: %s\n", cudaGetErrorString(error), context.c_str());
        throw runtime_error(cudaGetErrorString(error));
        // Handle error appropriately
    }
}

void checkCudaErrorCont(string context){
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\nAt: %s\n", cudaGetErrorString(error), context.c_str());
    }
}

void log(int verbose, int t, Electron* electrons_host, Electron* electrons, int* n_host, int* n, int capacity){
    if (verbose == 0 || t % verbose != 0) return;
    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    int true_n = min(*n_host, capacity);
    cudaMemcpy(electrons_host, electrons, true_n * sizeof(Electron), cudaMemcpyDeviceToHost);
    printf("Time %d, amount %d\n", t, *n_host);
    for(int i = 0; i < true_n; i++){
        electrons_host[i].print(i);
    }
    image(true_n, electrons_host, t); // visualize a snapshot of the current positions of the particles     
    printf("\n");

    checkCudaError("Log");
}

