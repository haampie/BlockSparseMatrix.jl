#include "stdint.h"
#include "immintrin.h"

extern "C" {

/**
 * Assume an n x m matrix A with blocks of size 2x2
 * Compute: Y = A * X + Y where we assume that
 * X and Y are of size n x 2
 * Uses 'naive' intrinstics. Compile with:
 * g++ bspmv.cc -o bspmv.so -shared -Wall -fPIC -O2 -march=native
 */
void bspmv2(int64_t n, int64_t * __restrict__ colptr, int64_t * __restrict__ rowval, 
            double * __restrict__ nzval, double * __restrict__ x, double * __restrict__ y)
{
    const int64_t size = 2 * n;
    double * A_ptr = nzval;
    double * x_ptr_1 = x;
    double * x_ptr_2 = x + size;
     

    __m128d x_reg_11, x_reg_21, x_reg_12, x_reg_22, tmp1, tmp2, y_reg_1, y_reg_2;

    // For each column of A
    #pragma nounroll
    for(int64_t j = 0; j < n; ++j)
    {
        // Load the x-values
        x_reg_11 = _mm_load_pd1(x_ptr_1 + 0);
        x_reg_21 = _mm_load_pd1(x_ptr_1 + 1);
        x_reg_12 = _mm_load_pd1(x_ptr_2 + 0);
        x_reg_22 = _mm_load_pd1(x_ptr_2 + 1);

        // Loop over each element in the column
        #pragma nounroll
        for (int64_t i = colptr[j] - 1; i < colptr[j + 1] - 1; ++i)
        {
            // Load the y-values
            double * y_ptr_1 = y + rowval[i] - 1;
            double * y_ptr_2 = y + rowval[i] + size - 1;

            // Load the first and second column of A
            tmp1 = _mm_load_pd(A_ptr);
            tmp2 = _mm_load_pd(A_ptr + 2);

            // Load the y values
            y_reg_1 = _mm_load_pd(y_ptr_1);
            y_reg_2 = _mm_load_pd(y_ptr_2);

            // y1 <- y1 .+ A[:, 1] .* x11
            // y1 <- y1 .+ A[:, 2] .* x21
            y_reg_1 = _mm_fmadd_pd(tmp1, x_reg_11, y_reg_1);
            y_reg_1 = _mm_fmadd_pd(tmp2, x_reg_21, y_reg_1);

            // store first column of y
            _mm_store_pd(y_ptr_1, y_reg_1);

            // y2 <- y2 .+ A[:, 1] .* x12
            // y2 <- y2 .+ A[:, 2] .* x22
            y_reg_2 = _mm_fmadd_pd(tmp1, x_reg_12, y_reg_2);
            y_reg_2 = _mm_fmadd_pd(tmp2, x_reg_22, y_reg_2);

            _mm_store_pd(y_ptr_2, y_reg_2);

            A_ptr += 4;
        }

        x_ptr_1 += 2;
        x_ptr_2 += 2;
    }
}
}