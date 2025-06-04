#include "read_file.h"
#include "MLP.h"
#include "activation.h"
#include "CNN.h"
#include "matrix.h"
#include "network.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define LEARNING_RATE 0.001f
#define EPOCHS 5
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define POOLING_KERNEL_SIZE 2
#define POOLING_STRIDE 2
#define KERNELS_NUM 3
#define CONVOLUTIONAL_KERNEL_SIZE 3
#define CONV1_PADDING 0
#define CONV1_STRIDE 1

#define DENSE2_SIZE 128
#define OUPUT_SIZE 10

#define TRAIN_IMG_PATH "data/train-images-idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels-idx1-ubyte"

int main() {
    int conv_output_size = (IMAGE_SIZE - CONVOLUTIONAL_KERNEL_SIZE + 2 * CONV1_PADDING) / CONV1_STRIDE + 1;
    int pooling_output_size = (conv_output_size - POOLING_KERNEL_SIZE) / POOLING_STRIDE + 1;
    int dense1_size = KERNELS_NUM * pooling_output_size * pooling_output_size;


    Network net;
    InputData data = {0};
    float learning_rate = LEARNING_RATE, img[INPUT_SIZE];
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));

    init_convolutional_layer(&net.conv_layer, KERNELS_NUM, CONVOLUTIONAL_KERNEL_SIZE, IMAGE_SIZE, CONV1_PADDING, CONV1_STRIDE);
    net.pooling_layer = malloc(KERNELS_NUM * sizeof(Max_Pooling_Layer));
    for (int i = 0; i < KERNELS_NUM; ++i) {
        init_pooling_layer(&net.pooling_layer[i], conv_output_size, POOLING_KERNEL_SIZE, POOLING_STRIDE);
    }
    init_dense_layer(&net.dense1_layer, dense1_size, DENSE2_SIZE);
    init_dense_layer(&net.output_layer, DENSE2_SIZE, OUPUT_SIZE);
    
    read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages, IMAGE_SIZE);
    read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

    shuffle_data(data.images, data.labels, data.nImages, INPUT_SIZE);

    //int train_size = (int)(data.nImages * TRAIN_SPLIT);
    int train_size = 10000;
    //int test_size = data.nImages - train_size;
    int test_size = 1000;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        start = clock();
        float total_loss = 0;

        Vector input_vector;

        for (int i = 0; i < train_size; i++) {
            fflush(stdout);
            for (int k = 0; k < INPUT_SIZE; k++) {
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            }

            input_vector.vector = img;
            input_vector.vector_size = INPUT_SIZE;
            float* final_output = train(&net, input_vector, data.labels[i], learning_rate);
            total_loss += cross_entropy_loss(final_output, data.labels[i]);
        }
        int correct = 0;
        //for (int i = train_size; i < data.nImages; i++) {
        for (int i = train_size; i < train_size + test_size; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            if (predict(net, input_vector) == data.labels[i])
                correct++;
        }
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", 
               epoch + 1, (float)correct / test_size * 100, total_loss / train_size, cpu_time_used);
    }
    free_convolutional_layer(&net.conv_layer);
    for (int i = 0; i < KERNELS_NUM; ++i) {
        free_max_pooling_layer(&net.pooling_layer[i]);
    }
    free(net.pooling_layer);
    free_dense_layer(&net.dense1_layer);
    free_dense_layer(&net.output_layer);
    free(data.images);
    free(data.labels);

    return 0;
}

