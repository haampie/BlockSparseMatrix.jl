# Block version of SparseMatrixCSC

Main idea: we divide the matrix A in _block elements_ of size 2x2: 

```
+--+--+
|11|33|
|11|33|
+--+--+
|22|44|
|22|44|
+--+--+
```
The `nzval` array will always store cells of `4` floating point numbers -- even when there
is a nonzero within a cell. This makes it possible to use SSE intrinsics when doing a sparse
matrix-vector multiplication. Obviously the `nzval` array can be 1 to 4 times as large
compared to the `SparseMatrixCSC` version. The `colptr` array will always be twice as small,
and the `rowval` can be 1 to 2 times as small.

The main motivation is to trade additional computational work for SSE instructions. Since
the sparse matrix-vector product is often memory-bound, this is perfectly feasible.

The routine in pseudo-code reads:

```
for j = 1 : n
  load X <- x[2j-1 : 2j]
  for i = colptr[j] : colptr[j+1]-1
    row = rowval[i]
    load Y <- y[2row-1 : 2row]
    load A <- reshape(nzval[i : i+3], 2, 2)
    y[2row-1 : 2row] <- Y + A * X # Here are the flops
  end
end
```

We have to optimize `Y + A * X`. In the case of `Float64`, we can pack things perfectly in
a 128-bit registers:

```
# Outer loop:
X1 = load [x1, x1]      # vmovddup
X2 = load [x2, x2]      # vmovddup
```

```
# Inner loop:
Y = load [y1, y2]
tmp1 = load [a11, a12]
tmp1 = tmp1 * X1 + Y    #vfmadd
tmp2 = load [a21, a22]
tmp2 = tmp2 * X2 + Y    #vfmadd
store Y in y
```
# Results

Currently I've implemented the routine `Y <- Y + A * X` where X and Y are vectors or
matrices of size `n x 2`. Benchmarks were run on a Macbook Air (Early 2015).

The first benchmark result is the BlockSparseMatrixCSC, second is SparseMatrixCSC. The 
relative storage costs are shown as well: `B_size / A_size` means the total storage costs
of the block format over the storage costs for SparseMatrixCSC.

## Tridiagonal matrix of order 1_000_000

First with `x` and `y` vectors:

```julia
> block, scalar = benchmark_banded(1_000_000, 1)
length(B.nzval) / length(A.nzval) = 1.9999986666657779 # Twice as many values stored
length(B.colptr) / length(A.colptr) = 0.5000004999995
length(B.rowval) / length(A.rowval) = 0.49999966666644446
B_size / A_size = 1.1428563469384343 # Slightly larger total storage costs
(Trial(5.811 ms), Trial(8.393 ms))
```

Speedup of 1.4x.

Now with `x` and `y` skinny matrices of size 1_000_000 by 2:

```julia
> block, scalar = benchmark2_banded(1_000_000, 1)
length(B.nzval) / length(A.nzval) = 1.9999986666657779
length(B.colptr) / length(A.colptr) = 0.5000004999995
length(B.rowval) / length(A.rowval) = 0.49999966666644446
B_size / A_size = 1.1428563469384343
(Trial(7.456 ms), Trial(15.718 ms))
```

Speedup of 2.1x.

## Pentadiagonal matrix of order 1_000_000

With `x` and `y` vectors:

```julia
> block, scalar = benchmark_banded(1_000_000, 2)
length(B.nzval) / length(A.nzval) = 1.199999839999808
length(B.colptr) / length(A.colptr) = 0.5000004999995
length(B.rowval) / length(A.rowval) = 0.299999959999952
B_size / A_size = 0.7272726363635454 # More efficient storage format!
(Trial(5.797 ms), Trial(13.323 ms))
```

Speedup of 2.2x.

With `x` and `y` matrices of size 1_000_000 by 2:

```julia
> block, scalar = benchmark2_banded(1_000_000, 2)
length(B.nzval) / length(A.nzval) = 1.199999839999808
length(B.colptr) / length(A.colptr) = 0.5000004999995
length(B.rowval) / length(A.rowval) = 0.299999959999952
B_size / A_size = 0.7272726363635454
(Trial(7.454 ms), Trial(23.956 ms))
```

Speedup of 3.2x.

## Random matrix of order 1_000_000 with 3 nonzeros per row

Matrices with a random sparsity pattern store a lot of structural zeros, so they serve as
a worst-case test case:

```julia
> benchmark_random(1_000_000, 3)
length(B.nzval) / length(A.nzval) = 3.9999826554215985 # We store 4x as many values
length(B.colptr) / length(A.colptr) = 0.5000004999995
length(B.rowval) / length(A.rowval) = 0.9999956638553996
B_size / A_size = 2.2141173860734913 # In the end we need ~2x as much memory
(Trial(44.003 ms), Trial(42.215 ms))
```

Speedup of 0.96x.

Now with `x` and `y` matrices of size 1_000_000 by 2:

```julia
> benchmark2_random(1_000_000, 3)
length(B.nzval) / length(A.nzval) = 3.9999906660350684
length(B.colptr) / length(A.colptr) = 0.5000004999995
length(B.rowval) / length(A.rowval) = 0.9999976665087671
B_size / A_size = 2.214263968129585
(Trial(100.798 ms), Trial(91.912 ms))
```

Speedup of 0.91x.