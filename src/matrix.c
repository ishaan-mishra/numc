#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high) / 1);
        }
    }
}



/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    int num_cols = mat->cols;
    int index = row * num_cols + col;
    return mat->data[index];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    int num_cols = mat->cols;
    int index = row * num_cols + col;
    mat->data[index] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix* mat_struct = malloc(sizeof(matrix));
    if (mat_struct == NULL) {
        return -2;
    }
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    mat_struct->data = calloc(rows * cols, sizeof(double));
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    mat_struct->rows = rows;
    mat_struct->cols = cols;
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    mat_struct->parent = NULL;
    // 6. Set the `ref_cnt` field to 1.
    mat_struct->ref_cnt = 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = mat_struct;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    if (mat == NULL) {
        return;
    }
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
     if (rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix* sliced_mat = malloc(sizeof(matrix));
    if (sliced_mat == NULL) {
        return -2;
    };
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    sliced_mat->data = from->data + offset;
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    sliced_mat->rows = rows;
    sliced_mat->cols = cols;
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    sliced_mat->parent = from;
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.;
    from->ref_cnt += 1;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = sliced_mat;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    int len = (mat->rows * mat->cols) / 16 * 16;
    __m256d val_vec = _mm256_set1_pd(val);
    #pragma omp parallel for
    for (unsigned int i = 0; i < len; i += 16) {
        _mm256_storeu_pd((mat->data + i), val_vec);
        _mm256_storeu_pd((mat->data + i + 4), val_vec);
        _mm256_storeu_pd((mat->data + i + 8), val_vec);
        _mm256_storeu_pd((mat->data + i + 12), val_vec);
    }
    // tail case
    #pragma omp parallel for
    for (unsigned int i = len; i < (mat->rows * mat->cols); i += 1) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int len = (mat->rows * mat->cols) / 16 * 16;
    __m256d _neg1 = _mm256_set1_pd(-1.0);
    __m256d tmp, tmp_neg, tmp_abs;
    #pragma omp parallel for private(tmp, tmp_neg, tmp_abs)
    for (unsigned int i = 0; i < len; i += 16) {
        tmp = _mm256_loadu_pd((mat->data + i));
        tmp_neg = _mm256_mul_pd(tmp, _neg1);
        tmp_abs = _mm256_max_pd(tmp, tmp_neg);
        _mm256_storeu_pd((result->data + i), tmp_abs);

        tmp = _mm256_loadu_pd((mat->data + i + 4));
        tmp_neg = _mm256_mul_pd(tmp, _neg1);
        tmp_abs = _mm256_max_pd(tmp, tmp_neg);
        _mm256_storeu_pd((result->data + i + 4), tmp_abs);

        tmp = _mm256_loadu_pd((mat->data + i + 8));
        tmp_neg = _mm256_mul_pd(tmp, _neg1);
        tmp_abs = _mm256_max_pd(tmp, tmp_neg);
        _mm256_storeu_pd((result->data + i + 8), tmp_abs);

        tmp = _mm256_loadu_pd((mat->data + i + 12));
        tmp_neg = _mm256_mul_pd(tmp, _neg1);
        tmp_abs = _mm256_max_pd(tmp, tmp_neg);
        _mm256_storeu_pd((result->data + i + 12), tmp_abs);
    }
    // tail case
    #pragma omp parallel for
    for (unsigned int i = len; i < (mat->rows * mat->cols); i += 1) {
        result->data[i] = fabs(mat->data[i]);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    for (unsigned int i = 0; i < mat->rows * mat->cols; i += 1) {
        result->data[i] = -(mat->data[i]);
    }
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    int len = (mat1->rows * mat1->cols) / 16 * 16;
    __m256d t1, t2, tmp;
    #pragma omp parallel for private(t1, t2, tmp)
    for (unsigned int i = 0; i < len; i += 16) {
        t1 = _mm256_loadu_pd((mat1->data + i));
        t2 = _mm256_loadu_pd((mat2->data + i));
        tmp = _mm256_add_pd(t1, t2);
        _mm256_storeu_pd((result->data + i), tmp);

        t1 = _mm256_loadu_pd((mat1->data + i + 4));
        t2 = _mm256_loadu_pd((mat2->data + i + 4));
        tmp = _mm256_add_pd(t1, t2);
        _mm256_storeu_pd((result->data + i + 4), tmp);

        t1 = _mm256_loadu_pd((mat1->data + i + 8));
        t2 = _mm256_loadu_pd((mat2->data + i + 8));
        tmp = _mm256_add_pd(t1, t2);
        _mm256_storeu_pd((result->data + i + 8), tmp);

        t1 = _mm256_loadu_pd((mat1->data + i + 12));
        t2 = _mm256_loadu_pd((mat2->data + i + 12));
        tmp = _mm256_add_pd(t1, t2);
        _mm256_storeu_pd((result->data + i + 12), tmp);
    }
    // tail case
    #pragma omp parallel for
    for (unsigned int i = len; i < (mat1->rows * mat1->cols); i += 1) {
        result->data[i] = mat1->data[i] + mat2->data[i]; 
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    for (unsigned int i = 0; i < mat1->rows * mat1->cols; i += 1) {
        result->data[i] = mat1->data[i] - mat2->data[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // matrix *mat2_t;
    // allocate_matrix(&mat2_t, mat2->cols, mat2->rows);
    // #pragma omp parallel for
    // for (int u = 0; u < mat2_t->rows; u += 1) {
    //     for (int v = 0; v < mat2_t->cols; v += 1) {
    //         mat2_t->data[u*mat2_t->cols + v] = mat2->data[v*mat2->cols + u];
    //     }
    // }

    int j, k;
    __m256d r;
    #pragma omp parallel for private(r, j, k)
    for (unsigned int i = 0; i < mat1->rows; i += 1) {
        for (j = 0; j < mat2->cols / 4 * 4; j += 4) {
            r = _mm256_setzero_pd(); //r = result[i][j]
            for (k = 0; k < mat1->cols; k += 1) {
                r = _mm256_fmadd_pd(
                    _mm256_broadcast_sd(mat1->data + i*mat1->cols + k), 
                    _mm256_loadu_pd(mat2->data + k*mat2->cols + j),
                    r);
            }
            _mm256_storeu_pd(result->data + i*result->cols + j, r);
        }
        // tail case
        for (unsigned int j = mat2->cols / 4 * 4; j < mat2->cols; j+= 1) {
           result->data[i*result->cols + j] = 0; 
           for (unsigned int k = 0; k < mat1->cols; k += 1) {
               result->data[i*result->cols + j] += mat1->data[i*mat1->cols + k] * mat2->data[k*mat2->cols + j];
           } 
        }
    }
    // deallocate_matrix(mat2_t);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int len = mat->rows * mat->cols * sizeof(double);
    if (pow == 0) {
        #pragma omp parallel for
        for (int i = 0; i < result->rows; i += 1) {
            for (int j = 0; j < result->cols; j += 1) {
                if (i == j) {
                    result->data[i*result->cols + j] = 1;
                } else {
                    result->data[i*result->cols + j] = 0;
                }
            }
        }
        return 0;
    }
    if (pow == 1) {
        memcpy(result->data, mat->data, len);
    }
    matrix* tmp;
    allocate_matrix(&tmp, mat->rows, mat->cols);
    mul_matrix(tmp, mat, mat);
    pow_matrix(result, tmp, pow/2);
    if (pow % 2 == 1) {
        memcpy(tmp->data, result->data, len);
        mul_matrix(result, tmp, mat);
    }
    deallocate_matrix(tmp);
    return 0;
}

// int mul_matrix_correct(matrix* result, matrix* mat1, matrix* mat2) {
//     matrix *mat2_t;
    // allocate_matrix(&mat2_t, mat2->cols, mat2->rows);
    // #pragma omp parallel for
    // for (int i = 0; i < mat2_t->rows; i += 1) {
    //     for (int j = 0; j < mat2_t->cols; j += 1) {
    //         mat2_t->data[i*mat2_t->cols + j] = mat2->data[j*mat2->cols + i];
    //     }
    // }

    // int j, k;
    // __m256d ra[4];
    // __m256d r;
    // #pragma omp parallel for private(r, ra, j, k)
    // for (unsigned int i = 0; i < mat1->rows; i += 1) {
    //     for (j = 0; j < mat2_t->rows / 16 * 16; j += 16) {
    //         for (int x=0; x < 4; x += 1) {
    //             ra[x] = _mm256_load_pd(result->data + i*result->cols + j + x*4);
    //         }
    //         for (k = 0; k < mat1->cols; k += 1) {
    //             __m256d m1 = _mm256_broadcast_sd(mat1->data + i*mat1->cols + k);
    //             for (int x=0; x < 4; x += 1) {
    //                 ra[x] = _mm256_fmadd_pd(
    //                     m1, 
    //                     _mm256_loadu_pd(mat2->data + k*mat2->cols + j + x*4),
    //                     ra[x]);
    //             }
    //         }
    //         for (int x=0; x < 4; x += 1) {
    //             _mm256_storeu_pd((result->data + i*result->cols + j + x*4), ra[x]);
    //         }
    //     }
    //     // tail case
    //     for (unsigned int j = mat2_t->rows / 16 * 16; j < mat2_t->rows / 4 * 4; j+= 4) {
    //         r = _mm256_setzero_pd(); //r = result[i][j]
    //         for (k = 0; k < mat1->cols; k += 1) {
    //             r = _mm256_fmadd_pd(
    //                 _mm256_broadcast_sd(mat1->data + i*mat1->cols + k), 
    //                 _mm256_loadu_pd(mat2->data + k*mat2->cols + j),
    //                 r);
    //         }
    //         _mm256_storeu_pd(result->data + i*result->cols + j, r);
    //     }

    //     for (unsigned int j = mat2_t->rows / 4 * 4; j < mat2_t->rows; j+= 1) {
    //        result->data[i*result->cols + j] = 0; 
    //        for (unsigned int k = 0; k < mat1->cols; k += 1) {
    //            result->data[i*result->cols + j] += mat1->data[i*mat1->cols + k] * mat2_t->data[j*mat2_t->cols + k];
    //        } 
    //     }
    // }
    // deallocate_matrix(mat2_t);
    // return 0;
// }
