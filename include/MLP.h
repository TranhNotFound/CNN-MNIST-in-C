#ifndef MLP_H
#define MLP_H

typedef struct {
    float **weights;
    float *biases;
    int input_size;
    int output_size;
} Layer;

void init_dense_layer(Layer *layer, int input_size, int output_size);
void forward(Layer *layer, float *input, float *output);
void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr);
void free_dense_layer(Layer *layer);

#endif // MLP_H