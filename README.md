# MNIST_CNN

This project implements a Convolutional Neural Network (CNN) from scratch in C for classifying handwritten digits from the MNIST dataset.

## Folder Structure

- **src/**: Source code files (`.c`) for all modules (CNN, MLP, matrix operations, file reading, etc.)
- **include/**: Header files (`.h`) for all modules.
- **data/**: Place MNIST dataset files here (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.).
- **Makefile**: Build script for compiling the project.
- **README.md**: Project description and usage instructions.

## Main Features

- **Custom CNN implementation**: Includes convolution, max pooling, dense layers, and activation functions. You can change network frame in network.h and corresponding function in network.c
- **Matrix and tensor operations**: All neural network computations are implemented manually.
- **MNIST data loading**: Functions to read and preprocess the MNIST dataset.
- **Training and evaluation**: Train the network and evaluate accuracy on test data.

## How to Build

1. Make sure you have GCC installed.
2. Create a `data/` folder and place the MNIST data files in it.
3. Open a terminal in the project directory and run:
    ```
    make
    ```
4. The executable `mnist_cnn` will be created in the project root.

## How to Run

After building, run:
```
./mnist_cnn
```
The program will train the CNN on the MNIST dataset and print training progress and accuracy.

## Performance

Test was performed in Network with 1 convolutional layer (3 kernels), max pooling layer, fully connected layer (128) and output layer (10).

Epoch 1, Accuracy: 86.90%, Avg Loss: 0.9637, Time: 15.69 seconds.
Epoch 2, Accuracy: 88.80%, Avg Loss: 0.3978, Time: 15.50 seconds.
Epoch 3, Accuracy: 89.70%, Avg Loss: 0.3252, Time: 15.50 seconds.
Epoch 4, Accuracy: 90.00%, Avg Loss: 0.2888, Time: 15.48 seconds.
Epoch 5, Accuracy: 90.60%, Avg Loss: 0.2636, Time: 15.41 seconds.

## Notes

- C language was not optimized for ML so configure training set and testing set small for fast training (I keep at 10000 and 1000 respectively).
- All memory management (malloc/free) is handled manually.
- The code is modularized for clarity and maintainability.
- You can adjust hyperparameters (epochs, learning rate, etc.) in `src/main.c`.

---

**Author:** Tranh
**Purpose:** Educational exercise for understanding and implementing convolutionnal neural networks in C.
