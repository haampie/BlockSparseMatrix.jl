#include "stdint.h"
#include "immintrin.h"

extern "C" {

/**
 * Assume an n x m matrix A with blocks of size 2x2
 * Compute: y = A * x + y
 * Uses 'naive' intrinstics. Compile with:
 * g++ bspmv.cc -o bspmv.so -shared -Wall -fPIC -O2 -march=native
 */
void bspmv(int64_t n, int64_t * __restrict__ colptr, int64_t * __restrict__ rowval, double * __restrict__ nzval, double * __restrict__ x, double * __restrict__ y)
{
    double * A_ptr = nzval;
    double * x_ptr = x;

    __m128d b1_reg, b2_reg, tmp1_reg, tmp2_reg, y_reg;

    // For each column of A
    #pragma nounroll
    for(int64_t j = 0; j < n; ++j)
    {
        // Load the x-values
        b1_reg = _mm_load_pd1(x_ptr);
        b2_reg = _mm_load_pd1(x_ptr + 1);

        // Loop over each element in the column
        #pragma nounroll
        for (int64_t i = colptr[j] - 1; i < colptr[j + 1] - 1; ++i)
        {
            double * y_ptr = y + rowval[i] - 1;

            // Load the first column of A
            tmp1_reg = _mm_load_pd(A_ptr);

            // Load the second column of A
            tmp2_reg = _mm_load_pd(A_ptr + 2);

            // Load the y values
            y_reg = _mm_load_pd(y_ptr);

            // y <- y .+ A[:, 1] .* x
            y_reg = _mm_fmadd_sd(tmp1_reg, b1_reg, y_reg);
            
            // y <- y .+ A[:, 2] .* x
            y_reg = _mm_fmadd_sd(tmp2_reg, b2_reg, y_reg);

            // store y
            _mm_store_pd(y_ptr, y_reg);

            A_ptr += 4;
        }

        x_ptr += 2;
    }
}
}