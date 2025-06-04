#include "network.h"
#include "MLP.h"
#include "CNN.h"
#include "activation.h"
#include "utils.h"
#include "matrix.h" // Ensure this header defines Square_Matrix and is used in convolution()
#include <stdio.h>
#include <stdlib.h>

// Ensure the Square_Matrix struct is included from the same header as used in convolution()
// If not, include or forward declare it here, but best practice is to include the correct header.

float *train(Network *net, Vector input, int label, float lr) {
    check_null(net, "Network is NULL");
    check_null(input.vector, "Input vector is not properly initialized");
    check_null(net->conv_layer.kernels.Matrix, "Kernel tensor is not properly initialized");
    check_null(net->conv_layer.biases.vector, "Bias vector is not properly initialized");

    check_null(net->pooling_layer, "Pooling layer is NULL");
    check_null(net->pooling_layer->mask.matrix, "Pooling layer mask matrix is not properly initialized");

    check_null(net->dense1_layer.weights, "Dense layer 1 weights are not properly initialized");
    check_null(net->dense1_layer.biases, "Dense layer 1 biases are not properly initialized");
    check_null(net->output_layer.weights, "Output layer weights are not properly initialized");
    check_null(net->output_layer.biases, "Output layer biases are not properly initialized");

    int kernels_num = net->conv_layer.kernels.depth;
    Square_Tensor3D conv_output;
    init_square_tensor3D(&conv_output, kernels_num, net->conv_layer.output_size);
    Square_Tensor3D pooling_output;
    init_square_tensor3D(&pooling_output, kernels_num, net->conv_layer.output_size / net->pooling_layer->stride);
    Vector flatten_channels;
    init_vector(&flatten_channels, kernels_num * (net->conv_layer.output_size / net->pooling_layer->stride) * (net->conv_layer.output_size / net->pooling_layer->stride));
    Vector dense1_output;
    init_vector(&dense1_output, net->dense1_layer.output_size);
    Vector final_output;
    init_vector(&final_output, net->output_layer.output_size);

    Square_Matrix pic;
    init_square_matrix(&pic, net->conv_layer.input_size);
    vector_to_matrix_reshape(input, &pic);

    for (int k = 0; k < kernels_num; ++k) {
        convolution(pic, net->conv_layer.kernels.Matrix[k], net->conv_layer.biases.vector[k], &conv_output.Matrix[k], net->conv_layer.padding, net->conv_layer.stride);
        matrix_relu(&conv_output.Matrix[k]);
        max_pooling(conv_output.Matrix[k], &pooling_output.Matrix[k], &net->pooling_layer[k].mask, net->pooling_layer[k].kernel_size, net->pooling_layer[k].stride);
    }
    tensor_to_vector_reshape(pooling_output, &flatten_channels);
    forward(&net->dense1_layer, flatten_channels.vector, dense1_output.vector);
    forward(&net->output_layer, dense1_output.vector, final_output.vector);
    softmax(final_output.vector, net->output_layer.output_size);

    //backprop
    Vector dflat;
    init_vector(&dflat, kernels_num * (net->conv_layer.output_size / net->pooling_layer->stride) * (net->conv_layer.output_size / net->pooling_layer->stride));
    Square_Tensor3D pool_grad;
    init_square_tensor3D(&pool_grad, kernels_num, net->conv_layer.output_size / net->pooling_layer->stride);
    Square_Tensor3D dconv_relu;
    init_square_tensor3D(&dconv_relu, kernels_num, net->conv_layer.output_size);
    Square_Tensor3D dx;
    init_square_tensor3D(&dx, kernels_num, net->conv_layer.input_size);
    Vector output_grad;
    init_vector(&output_grad, net->output_layer.output_size);
    Vector dense1_grad;
    init_vector(&dense1_grad, net->dense1_layer.output_size);
    for (int i = 0; i < net->output_layer.output_size; i++)
        output_grad.vector[i] = final_output.vector[i] - (i == label); // dZ3 (cross entropy loss)
    backward(&net->output_layer, dense1_output.vector, output_grad.vector, dense1_grad.vector, lr); // dW3, dB3

    for (int i = 0; i < net->dense1_layer.output_size; i++) // dZ2
        dense1_grad.vector[i] *= relu_derivative(dense1_output.vector[i]);
    backward(&net->dense1_layer, flatten_channels.vector, dense1_grad.vector, dflat.vector, lr); // dW2, dB2

    vector_to_tensor_reshape(dflat, &pool_grad);
    for (int k = 0; k < kernels_num; ++k) {
        max_pooling_backward(pool_grad.Matrix[k], &dconv_relu.Matrix[k], net->pooling_layer[k].mask, net->pooling_layer[k].kernel_size, net->pooling_layer[k].stride);

        for (int i = 0; i < net->conv_layer.output_size; ++i) {
            for (int j = 0; j < net->conv_layer.output_size; ++j) {
                dconv_relu.Matrix[k].matrix[i][j] *= relu_derivative(conv_output.Matrix[k].matrix[i][j]);
            }
        }
        convolution_backward(pic, dconv_relu.Matrix[k], net->conv_layer.kernels.Matrix[k], &dx.Matrix[k], &net->conv_layer.kernels.Matrix[k], &net->conv_layer.biases.vector[k], net->conv_layer.stride, lr);
    }
    free_square_tensor3D(&conv_output);
    free_square_tensor3D(&pooling_output);
    free_vector(&flatten_channels);
    free_vector(&dense1_output);
    free_vector(&output_grad);
    free_vector(&dense1_grad);
    free_vector(&dflat);
    free_square_tensor3D(&pool_grad);
    free_square_tensor3D(&dconv_relu);
    free_square_tensor3D(&dx);
    free_square_matrix(&pic);
    return final_output.vector;
}

int predict(Network net, Vector input) {
    int kernels_num = net.conv_layer.kernels.depth;
    Square_Tensor3D conv_output;
    init_square_tensor3D(&conv_output, kernels_num, net.conv_layer.output_size);
    Square_Tensor3D pooling_output;
    init_square_tensor3D(&pooling_output, kernels_num, net.conv_layer.output_size / net.pooling_layer->stride);
    Vector flatten_channels;
    init_vector(&flatten_channels, kernels_num * (net.conv_layer.output_size / net.pooling_layer->stride) * (net.conv_layer.output_size / net.pooling_layer->stride));
    Vector dense1_output;
    init_vector(&dense1_output, net.dense1_layer.output_size);
    Vector final_output;
    init_vector(&final_output, net.output_layer.output_size);

    Square_Matrix pic;
    init_square_matrix(&pic, net.conv_layer.input_size);
    vector_to_matrix_reshape(input, &pic);

    for (int k = 0; k < kernels_num; ++k) {
        convolution(pic, net.conv_layer.kernels.Matrix[k], net.conv_layer.biases.vector[k], &conv_output.Matrix[k], net.conv_layer.padding, net.conv_layer.stride);
        matrix_relu(&conv_output.Matrix[k]);
        max_pooling(conv_output.Matrix[k], &pooling_output.Matrix[k], &net.pooling_layer[k].mask, net.pooling_layer[k].kernel_size, net.pooling_layer[k].stride);

    }
    tensor_to_vector_reshape(pooling_output, &flatten_channels);
    forward(&net.dense1_layer, flatten_channels.vector, dense1_output.vector);
    forward(&net.output_layer, dense1_output.vector, final_output.vector);
    softmax(final_output.vector, net.output_layer.output_size);

    free_square_tensor3D(&conv_output);
    free_square_tensor3D(&pooling_output);
    free_vector(&flatten_channels);
    free_vector(&dense1_output);
    free_square_matrix(&pic);

    int max_index = 0;
    for (int i = 1; i < net.output_layer.output_size; i++)
        if (final_output.vector[i] > final_output.vector[max_index])
            max_index = i;

    return max_index;
}

