#ifndef activation_h
#define activation_h

#include "matrix.h"

float relu(float x);
void matrix_relu(Square_Matrix *matrix);
float relu_derivative(float x);
void matrix_relu_derivative(Square_Matrix *matrix);
void softmax(float *input, int size);
float cross_entropy_loss(float *predictions, int label);

#endif // activation_h