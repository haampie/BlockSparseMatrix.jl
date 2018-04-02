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
