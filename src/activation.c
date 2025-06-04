#include "activation.h"
#include "matrix.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float relu(float x) {
    return x > 0 ? x : 0;
}

void matrix_relu(Square_Matrix *matrix) {
    check_null(matrix, "Matrix is NULL");
    check_null(matrix->matrix, "Matrix is not properly initialized");
    for (int i = 0; i < matrix->matrix_size; ++i) {
        for (int j = 0; j < matrix->matrix_size; ++j) {
            matrix->matrix[i][j] = relu(matrix->matrix[i][j]);
        }
    }
}

float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

void matrix_relu_derivative(Square_Matrix *matrix) {
    check_null(matrix, "Matrix is NULL");
    check_null(matrix->matrix, "Matrix is not properly initialized");
    for (int i = 0; i < matrix->matrix_size; ++i) {
        for (int j = 0; j < matrix->matrix_size; ++j) {
            matrix->matrix[i][j] = relu_derivative(matrix->matrix[i][j]);
        }
    }
}

void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

float cross_entropy_loss(float *predictions, int label) {
    return -logf(predictions[label] + 1e-10f);
}