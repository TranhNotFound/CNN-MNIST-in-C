#ifndef CNN_H
#define CNN_H

#include "utils.h"
#include "matrix.h"

typedef struct {
    Square_Tensor3D kernels; // Mảng 3D chứa các kernel
    Vector biases;    // Mảng chứa các bias tương ứng với mỗi kernel
    int padding;     // Padding cho convolution
    int stride;      // Stride cho convolution
    int input_size;  // Kích thước đầu vào (chiều cao và chiều rộng)
    int output_size; // Kích thước đầu ra (chiều cao và chiều rộng)
} Convolutional_Layer;

typedef struct {
    Square_Matrix mask; // Mảng 2D chứa mask cho pooling
    int input_size; // Kích thước đầu vào (chiều cao và chiều rộng)
    int kernel_size; // Kích thước của kernel pooling
    int stride; // Stride cho pooling
} Max_Pooling_Layer;

void convolution(Square_Matrix input, Square_Matrix kernel, float bias, Square_Matrix *output, int padding, int stride);
void convolution_backward(Square_Matrix input, Square_Matrix output_grad, Square_Matrix filter, Square_Matrix *input_grad, Square_Matrix *kernel, float *bias, int stride, int lr);
void max_pooling(Square_Matrix input, Square_Matrix *output, Square_Matrix *mask, int kernel_size, int stride);
void max_pooling_backward(Square_Matrix output_grad, Square_Matrix *input_grad, Square_Matrix mask, int kernel_size, int stride);
void init_filter(Square_Matrix *filter);
void init_convolutional_layer(Convolutional_Layer *conv_layer, int num_kernels, int kernel_size, int input_size, int padding, int stride);
void init_pooling_layer(Max_Pooling_Layer *pooling_layer, int input_size, int kernel_size, int stride);
void free_convolutional_layer(Convolutional_Layer *conv_layer);
void free_max_pooling_layer(Max_Pooling_Layer *pooling_layer);

#endif // CNN_H