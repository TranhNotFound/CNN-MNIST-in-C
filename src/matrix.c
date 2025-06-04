#include "matrix.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void matrix_multiply(Square_Matrix a, Square_Matrix b, Square_Matrix *result) {
    check_null(result, "Result matrix is NULL");
    check_null(result->matrix, "Result matrix is not properly initialized");
    check_null(a.matrix, "Input matrix A is not properly initialized");
    check_null(b.matrix, "Input matrix B is not properly initialized");
    if (a.matrix_size != b.matrix_size) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        exit(EXIT_FAILURE);
    }
    if (result->matrix_size != a.matrix_size) {
        fprintf(stderr, "Result matrix size does not match input matrices.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < a.matrix_size; ++i) {
        for (int j = 0; j < b.matrix_size; ++j) {
            result->matrix[i][j] = 0;
            for (int k = 0; k < a.matrix_size; ++k) {
                result->matrix[i][j] += a.matrix[i][k] * b.matrix[k][j];
            }
        }
    }
}

float matrix_sum(Square_Matrix matrix) {
    float sum = 0.0f;
    for (int i = 0; i < matrix.matrix_size; ++i) {
        for (int j = 0; j < matrix.matrix_size; ++j) {
            sum += matrix.matrix[i][j];
        }
    }
    return sum;
}

void elementwise_calculate(Square_Matrix a, Square_Matrix b, Square_Matrix *result, char calculate_type) {
    check_null(result, "Result matrix is NULL");
    check_null(result->matrix, "Result matrix is not properly initialized");
    check_null(a.matrix, "Input matrix A is not properly initialized");
    check_null(b.matrix, "Input matrix B is not properly initialized");
    if (a.matrix_size != b.matrix_size) {
        fprintf(stderr, "Matrix dimensions do not match for elementwise calculation.\n");
        exit(EXIT_FAILURE);
    }
    if (calculate_type != '+' && calculate_type != '*' && calculate_type != '-') {
        fprintf(stderr, "Invalid calculation type. Use '+ - *' only.\n");
        exit(EXIT_FAILURE);
    }
    if (result->matrix_size != a.matrix_size) {
        fprintf(stderr, "Result matrix size does not match input matrices.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < a.matrix_size; ++i) {
        for (int j = 0; j < a.matrix_size; ++j) {
            switch (calculate_type) {
                case '+':
                    result->matrix[i][j] = a.matrix[i][j] + b.matrix[i][j];
                    break;
                case '-':
                    result->matrix[i][j] = a.matrix[i][j] - b.matrix[i][j];
                    break;
                case '*':
                    result->matrix[i][j] = a.matrix[i][j] * b.matrix[i][j];
                    break;
            }
        }
    }
}

void elementwise_multiply_num(Square_Matrix a, float m, Square_Matrix *result) {
    check_null(result, "Result matrix is NULL");
    check_null(result->matrix, "Result matrix is not properly initialized");
    check_null(a.matrix, "Input matrix is not properly initialized");
    if (result->matrix_size != a.matrix_size) {
        fprintf(stderr, "Result matrix size does not match input matrix size.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < a.matrix_size; ++i) {
        for (int j = 0; j < a.matrix_size; ++j) {
            result->matrix[i][j] = a.matrix[i][j] * m;
        }
    }
}

void region(Square_Matrix input, Square_Matrix *output, int region_size, int row_start, int col_start) {
    if (row_start < 0 || col_start < 0 || row_start + region_size > input.matrix_size || col_start + region_size > input.matrix_size) {
        fprintf(stderr, "Region out of bounds.\n");
        exit(EXIT_FAILURE);
    }
    check_null(output, "Output matrix is NULL");
    check_null(input.matrix, "Input matrix is not properly initialized");
    check_null(output->matrix, "Output matrix is not properly initialized");
    if (output->matrix_size != region_size) {
        fprintf(stderr, "Output matrix size does not match region size.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < region_size; ++i) {
        for (int j = 0; j < region_size; ++j) {
            output->matrix[i][j] = input.matrix[row_start + i][col_start + j];
        }
    }
}


void init_square_matrix(Square_Matrix *matrix, int matrix_size) {
    check_null(matrix, "Matrix structure is NULL");
    matrix->matrix_size = matrix_size;
    matrix->matrix = malloc(matrix_size * sizeof(float*));
    check_null(matrix->matrix, "Failed to allocate memory for matrix rows");

    for (int i = 0; i < matrix_size; ++i) {
        matrix->matrix[i] = calloc(matrix_size, sizeof(float));
        if (matrix->matrix[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for matrix row %d\n", i);
            // Free any previously allocated rows before exit
            for (int j = 0; j < i; ++j) {
                free(matrix->matrix[j]);
            }
            free(matrix->matrix);
            exit(EXIT_FAILURE);
        }
    }
}

void free_square_matrix(Square_Matrix *matrix) {
    check_null(matrix, "Matrix structure is freed");
    check_null(matrix->matrix, "Matrix is freed");
    for (int i = 0; i < matrix->matrix_size; ++i) {
        free(matrix->matrix[i]);
    }
    free(matrix->matrix);
    matrix->matrix = NULL; // Avoid dangling pointer
}

float max_matrix_value(Square_Matrix matrix, int *max_index_i, int *max_index_j) {
    check_null(matrix.matrix, "Matrix is not properly initialized");
    float max_value = matrix.matrix[0][0];
    *max_index_i = 0;
    *max_index_j = 0;
    for (int i = 0; i < matrix.matrix_size; ++i) {
        for (int j = 0; j < matrix.matrix_size; ++j) {
            if (matrix.matrix[i][j] > max_value) {
                max_value = matrix.matrix[i][j];
                *max_index_i = i;
                *max_index_j = j;
            }
        }
    }
    return max_value;
}

float sum_matrix(Square_Matrix matrix) {
    check_null(matrix.matrix, "Matrix is not properly initialized");
    float sum = 0.0f;
    for (int i = 0; i < matrix.matrix_size; ++i) {
        for (int j = 0; j < matrix.matrix_size; ++j) {
            sum += matrix.matrix[i][j];
        }
    }
    return sum;
}

void padding_add(Square_Matrix input, int padding, Square_Matrix *output) {
    check_null(input.matrix, "Input matrix is not properly initialized");
    check_null(output, "Output matrix is NULL");
    check_null(output->matrix, "Output matrix is not properly initialized");

    int input_size = input.matrix_size;
    int output_size = input_size + 2 * padding;
    if (output->matrix_size != output_size) {
        fprintf(stderr, "Output matrix size does not match expected size after padding.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            output->matrix[i + padding][j + padding] = input.matrix[i][j];
        }
    }
}

void square_matrix_rotate(Square_Matrix matrix, Square_Matrix *rotated_matrix) {
    check_null(matrix.matrix, "Input matrix is not properly initialized");
    check_null(rotated_matrix, "Rotated matrix is NULL");
    check_null(rotated_matrix->matrix, "Rotated matrix is not properly initialized");

    int matrix_size = matrix.matrix_size;
    if (rotated_matrix->matrix_size != matrix_size) {
        fprintf(stderr, "Rotated matrix size does not match input matrix size.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            rotated_matrix->matrix[j][matrix_size - 1 - i] = matrix.matrix[i][j];
        }
    }
}

void vector_to_matrix_reshape(Vector vector, Square_Matrix *matrix) {
    check_null(vector.vector, "Vector is not properly initialized");
    check_null(matrix, "Picture matrix is NULL");
    check_null(matrix->matrix, "Picture matrix is not properly initialized");

    int matrix_size = matrix->matrix_size;
    if (vector.vector_size != matrix_size * matrix_size) {
        fprintf(stderr, "Matrix size does not match vector size.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            matrix->matrix[i][j] = vector.vector[i * matrix_size + j];
        }
    }
}

void vector_to_tensor_reshape(Vector vector, Square_Tensor3D *tensor) {
    check_null(vector.vector, "Vector is not properly initialized");
    check_null(tensor, "Tensor is NULL");
    check_null(tensor->Matrix, "Tensor is not properly initialized");

    int depth = tensor->depth;
    int square_size = tensor->Matrix->matrix_size;
    if (vector.vector_size != depth * square_size * square_size) {
        fprintf(stderr, "Tensor size does not match vector size.\n");
        exit(EXIT_FAILURE);
    }

    for (int k = 0; k < depth; ++k) {
        for (int i = 0; i < square_size; ++i) {
            for (int j = 0; j < square_size; ++j) {
                (*tensor).Matrix[k].matrix[i][j] = vector.vector[k * square_size * square_size + i * square_size + j];
            }
        }
    }
}

void tensor_to_vector_reshape(Square_Tensor3D tensor, Vector *vector) {
    for (int k = 0; k < tensor.depth; ++k) {
        check_null(tensor.Matrix[k].matrix, "Tensor matrix is not properly initialized");
    }
    check_null(vector, "Vector is NULL");
    check_null(vector->vector, "Vector is not properly initialized");

    int depth = tensor.depth;
    int square_size = tensor.Matrix[0].matrix_size;
    if (vector->vector_size != depth * square_size * square_size) {
        fprintf(stderr, "Vector size does not match tensor size.\n");
        exit(EXIT_FAILURE);
    }

    for (int k = 0; k < depth; ++k) {
        for (int i = 0; i < square_size; ++i) {
            for (int j = 0; j < square_size; ++j) {
                vector->vector[k * square_size * square_size + i * square_size + j] = tensor.Matrix[k].matrix[i][j];
            }
        }
    }
}

void init_vector(Vector *vector, int vector_size) {
    check_null(vector, "Vector structure is NULL");
    vector->vector_size = vector_size;
    vector->vector = malloc(vector_size * sizeof(float));
    check_null(vector->vector, "Failed to allocate memory for vector");
    for (int i = 0; i < vector_size; ++i) {
        vector->vector[i] = 0.0f; // Initialize to zero
    }
}

void free_vector(Vector *vector) {
    check_null(vector, "Vector structure is freed");
    check_null(vector->vector, "Vector is freed");
    free(vector->vector);
    vector->vector = NULL; // Avoid dangling pointer
}

void init_square_tensor3D(Square_Tensor3D *tensor, int depth, int matrix_size) {
    check_null(tensor, "Tensor structure is NULL");
    tensor->depth = depth;
    tensor->Matrix = malloc(depth * sizeof(Square_Matrix));
    check_null(tensor->Matrix, "Failed to allocate memory for tensor matrices");

    for (int i = 0; i < depth; ++i) {
        init_square_matrix(&tensor->Matrix[i], matrix_size);
    }
}

void free_square_tensor3D(Square_Tensor3D *tensor) {
    check_null(tensor, "Tensor structure is freed");
    check_null(tensor->Matrix, "Tensor matrices are freed");
    for (int i = 0; i < tensor->depth; ++i) {
        free_square_matrix(&tensor->Matrix[i]);
    }
    free(tensor->Matrix);
    tensor->Matrix = NULL; // Avoid dangling pointer
}