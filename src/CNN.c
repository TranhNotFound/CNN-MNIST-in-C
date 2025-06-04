#include "CNN.h"
#include "activation.h"
#include "matrix.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void convolution(Square_Matrix input, Square_Matrix kernel, float bias, Square_Matrix *output, int padding, int stride) {
    check_null(input.matrix, "Input matrix is not properly initialized");
    check_null(kernel.matrix, "Kernel matrix is not properly initialized");
    check_null(output, "Output matrix is NULL");
    check_null(output->matrix, "Output matrix is not properly initialized");

    int input_size = input.matrix_size;
    int kernel_size = kernel.matrix_size;
    int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    if (output->matrix_size != output_size) {
        fprintf(stderr, "Output matrix size does not match expected size.\n");
        exit(EXIT_FAILURE);
    }

    Square_Matrix padded_input;
    int padded_input_size = input_size + 2 * padding;
    init_square_matrix(&padded_input, padded_input_size);
    padding_add(input, padding, &padded_input);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            Square_Matrix region_matrix;
            init_square_matrix(&region_matrix, kernel_size);
            region(padded_input, &region_matrix, kernel_size, i * stride - padding, j * stride - padding);
            elementwise_calculate(region_matrix, kernel, &region_matrix, '*');
            output->matrix[i][j] = matrix_sum(region_matrix) + bias;
            free_square_matrix(&region_matrix);
        }
    }
    free_square_matrix(&padded_input);
}

void convolution_backward(Square_Matrix input, Square_Matrix output_grad, Square_Matrix filter, Square_Matrix *input_grad, Square_Matrix *kernel, float *bias, int stride, int lr) {
    check_null(input.matrix, "Input matrix is not properly initialized");
    check_null(output_grad.matrix, "Output gradient matrix is not properly initialized");
    check_null(input_grad, "Input gradient matrix is NULL");
    check_null(input_grad->matrix, "Input gradient matrix is not properly initialized");
    check_null(kernel, "Kernel gradient matrix is NULL");
    check_null(kernel->matrix, "Kernel gradient matrix is not properly initialized");

    int input_size = input.matrix_size;
    int kernel_size = kernel->matrix_size;
    int output_size = output_grad.matrix_size;

    if (input_grad->matrix_size != input_size) {
        fprintf(stderr, "Input gradient matrix size does not match input size.\n");
        exit(EXIT_FAILURE);
    }

    int full_padding = (kernel_size - 1) / 2;
    Square_Matrix padded_input;
    int padded_input_size = input_size + 2 * full_padding;
    init_square_matrix(&padded_input, padded_input_size);
    padding_add(input, full_padding, &padded_input);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            Square_Matrix region_matrix;
            init_square_matrix(&region_matrix, kernel_size);
            region(padded_input, &region_matrix, kernel_size, i * stride, j * stride);
            elementwise_multiply_num(region_matrix, lr * output_grad.matrix[i][j], &region_matrix);
            elementwise_calculate(*kernel, region_matrix, kernel, '-');
            free_square_matrix(&region_matrix);
        }
    }
    Square_Matrix rotated_filter;
    init_square_matrix(&rotated_filter, kernel_size);
    square_matrix_rotate(filter, &rotated_filter);
    convolution(padded_input, rotated_filter, 0.0f, input_grad, 0, stride);
    free_square_matrix(&rotated_filter);
    free_square_matrix(&padded_input);
    
    *bias -= lr * sum_matrix(output_grad); // Update bias
}

void max_pooling(Square_Matrix input, Square_Matrix *output, Square_Matrix *mask, int kernel_size, int stride) {
    check_null(input.matrix, "Input matrix is not properly initialized");
    check_null(output, "Output matrix is NULL");
    check_null(output->matrix, "Output matrix is not properly initialized");
    check_null(mask, "Mask matrix is NULL");
    check_null(mask->matrix, "Mask matrix is not properly initialized");

    int input_size = input.matrix_size;
    int output_size = (input_size - kernel_size) / stride + 1;

    if (output->matrix_size != output_size) {
        //printf("Maxpooling.Output matrix size: %d, Expected size: %d\n", output->matrix_size, output_size);
        fprintf(stderr, "Output matrix size does not match expected size.\n");
        exit(EXIT_FAILURE);
    }
    if (mask->matrix_size != input_size) {
        fprintf(stderr, "Mask matrix size does not match input size.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            int max_index_i = -1, max_index_j = -1;
            Square_Matrix region_matrix;
            init_square_matrix(&region_matrix, kernel_size);
            region(input, &region_matrix, kernel_size, i * stride, j * stride);
            output->matrix[i][j] = max_matrix_value(region_matrix, &max_index_i, &max_index_j);
            mask->matrix[i * stride + max_index_i][j * stride + max_index_j] = 1.0f; // Store the mask
            free_square_matrix(&region_matrix);
        }
    }
}

void max_pooling_backward(Square_Matrix output_grad, Square_Matrix *input_grad, Square_Matrix mask, int kernel_size, int stride) {
    check_null(input_grad, "Input gradient matrix is NULL");
    check_null(input_grad->matrix, "Input gradient matrix is not properly initialized");
    check_null(mask.matrix, "Mask matrix is not properly initialized");
    check_null(output_grad.matrix, "Output gradient matrix is not properly initialized");

    int output_size = output_grad.matrix_size;
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    if (mask.matrix[i * stride + ki][j * stride + kj] == 1.0f) {
                        input_grad->matrix[i * stride + ki][j * stride + kj] += output_grad.matrix[i][j];
                    }
                }
            }
        }
    }
}

void init_filter(Square_Matrix *filter) {
    check_null(filter, "Filter is NULL");
    check_null(filter->matrix, "Filter matrix is not properly initialized");
    for (int i = 0; i < filter->matrix_size; ++i) {
        for (int j = 0; j < filter->matrix_size; ++j) {
            filter->matrix[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrt(2.0f / filter->matrix_size); // Random initialization
        }
    }
}
void init_convolutional_layer(Convolutional_Layer *conv_layer, int num_kernels, int kernel_size, int input_size, int padding, int stride) {
    conv_layer->padding = padding;
    conv_layer->stride = stride;
    conv_layer->input_size = input_size;
    conv_layer->output_size = (input_size - kernel_size + 2 * padding) / stride + 1;

    conv_layer->kernels.depth = num_kernels;
    conv_layer->kernels.Matrix = malloc(num_kernels * sizeof(Square_Matrix));
    check_null(conv_layer->kernels.Matrix, "Kernel tensor is not properly initialized");

    for (int k = 0; k < num_kernels; ++k) {
        init_square_matrix(&conv_layer->kernels.Matrix[k], kernel_size);
        init_filter(&conv_layer->kernels.Matrix[k]);
    }
    conv_layer->biases.vector_size = num_kernels;
    conv_layer->biases.vector = calloc(num_kernels, sizeof(float));
    check_null(conv_layer->biases.vector, "Bias vector is not properly initialized");
}

void init_pooling_layer(Max_Pooling_Layer *pooling_layer, int input_size, int kernel_size, int stride) {
    pooling_layer->input_size = input_size;
    pooling_layer->kernel_size = kernel_size;
    pooling_layer->stride = stride;

    init_square_matrix(&pooling_layer->mask, input_size);
}

void free_convolutional_layer(Convolutional_Layer *conv_layer) {
    check_null(conv_layer, "Convolutional layer is NULL");
    for (int k = 0; k < conv_layer->kernels.depth; ++k) {
        free_square_matrix(&conv_layer->kernels.Matrix[k]);
    }
    free(conv_layer->kernels.Matrix);
    free(conv_layer->biases.vector);
}

void free_max_pooling_layer(Max_Pooling_Layer *pooling_layer) {
    check_null(pooling_layer, "Pooling layer is NULL");
    free_square_matrix(&pooling_layer->mask);
}