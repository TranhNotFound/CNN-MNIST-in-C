#include "read_file.h"
#include <stdio.h>
#include <stdlib.h>

void read_mnist_images(const char *filename, unsigned char **images, int *nImages, int image_size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }

    int temp, rows, cols;
    fread(&temp, sizeof(int), 1, file);
    fread(nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    *images = malloc((*nImages) * image_size * image_size);
    fread(*images, sizeof(unsigned char), (*nImages) * image_size * image_size, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Cannot open file: %s\n", filename);
        exit(1);
    }

    int temp;
    fread(&temp, sizeof(int), 1, file);
    fread(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);
    fread(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n, int input_size) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1); // random index 0-i
        // Swap images
        for (int k = 0; k < input_size; k++) {
            unsigned char temp = images[i * input_size + k];
            images[i * input_size + k] = images[j * input_size + k];
            images[j * input_size + k] = temp;
        }
        // Swap labels
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}