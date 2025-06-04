#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float **matrix;
    int matrix_size;
} Square_Matrix;

typedef struct {
    float *vector;
    int vector_size;
}  Vector;

typedef struct {
    Square_Matrix *Matrix; // Mảng 3D chứa các ma trận vuông
    int depth;
} Square_Tensor3D;

void matrix_multiply(Square_Matrix a, Square_Matrix b, Square_Matrix *result);
float matrix_sum(Square_Matrix matrix);
void elementwise_calculate(Square_Matrix a, Square_Matrix b, Square_Matrix *result, char calculate_type);
void elementwise_multiply_num(Square_Matrix a, float m, Square_Matrix *result);
void region(Square_Matrix input, Square_Matrix *output, int region_size, int row_start, int col_start);
void init_square_matrix(Square_Matrix *matrix, int matrix_size);
void free_square_matrix(Square_Matrix *matrix);
float max_matrix_value(Square_Matrix matrix, int *max_index_i, int *max_index_j);
float sum_matrix(Square_Matrix matrix);
void padding_add(Square_Matrix input, int padding, Square_Matrix *output);
void square_matrix_rotate(Square_Matrix matrix, Square_Matrix *rotated_matrix);
void vector_to_matrix_reshape(Vector vector, Square_Matrix *matrix);
void vector_to_tensor_reshape(Vector vector, Square_Tensor3D *tensor);
void tensor_to_vector_reshape(Square_Tensor3D tensor, Vector *vector);
void init_vector(Vector *vector, int vector_size);
void free_vector(Vector *vector);
void init_square_tensor3D(Square_Tensor3D *tensor, int depth, int matrix_size);
void free_square_tensor3D(Square_Tensor3D *tensor);

#endif
