#ifndef read_file_h
#define read_file_h

typedef struct {
    unsigned char *images, *labels;
    int nImages;
} InputData;


void read_mnist_images(const char *filename, unsigned char **images, int *nImages, int image_size);
void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels);
void shuffle_data(unsigned char *images, unsigned char *labels, int n, int input_size);

#endif // read_file_h