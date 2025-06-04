#include "MLP.h"
#include "activation.h"
#include "utils.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void init_dense_layer(Layer *layer, int input_size, int output_size) {
    float scale = sqrtf(2.0f / input_size);
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = malloc(input_size * sizeof(float*));
    check_null(layer->weights, "Weights matrix is not properly initialized");
    layer->biases = calloc(output_size, sizeof(float));
    check_null(layer->biases, "Biases vector is not properly initialized");
    for (int i = 0; i < input_size; ++i) {
        layer->weights[i] = malloc(output_size * sizeof(float));
        for (int j = 0; j < output_size; ++j) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale; // Random initialization
        }
    }
}

void forward(Layer *layer, float *input, float *output) {
    for (int j = 0; j < layer->output_size; ++j) {
        output[j] = layer->biases[j];
        for (int i = 0; i < layer->input_size; ++i) {
            output[j] += input[i] * layer->weights[i][j];
        }
    }
    for (int i = 0; i < layer->output_size; ++i) {
        output[i] = relu(output[i]);
    }
}

void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    // 1. Tính gradient lan truyền về đầu vào (trừ layer đầu vào)
    if (input_grad) {
        for (int i = 0; i < layer->input_size; ++i) {
            input_grad[i] = 0.0f;
            for (int j = 0; j < layer->output_size; ++j) {
                input_grad[i] += output_grad[j] * layer->weights[i][j];
            }
        }
    }
    // 2. Cập nhật trọng số
    for (int i = 0; i < layer->input_size; ++i) {
        for (int j = 0; j < layer->output_size; ++j) {
            float grad = output_grad[j] * input[i]; // dW = dZ * A
            layer->weights[i][j] -= lr * grad;
        }
    }
    // 3. Cập nhật bias
    for (int i = 0; i < layer->output_size; i++) {
        layer->biases[i] -= lr * output_grad[i];
    }
}

void free_dense_layer(Layer *layer) {
    check_null(layer, "Layer is NULL");
    for (int i = 0; i < layer->input_size; ++i) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    layer->weights = NULL;
    layer->biases = NULL;
}