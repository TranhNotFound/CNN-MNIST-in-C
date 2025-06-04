#ifndef network_h
#define network_h

#include "MLP.h"
#include "matrix.h"
#include "CNN.h"

typedef struct {
    Convolutional_Layer conv_layer;
    Max_Pooling_Layer *pooling_layer;
    Layer dense1_layer;
    Layer output_layer;
} Network;

float *train(Network *net, Vector input, int label, float lr);
int predict(Network net, Vector input);

#endif // network_h