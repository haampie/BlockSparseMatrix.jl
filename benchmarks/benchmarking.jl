module BenchBlockSparseMatrix

using BenchmarkTools, BlockSparseMatrix

banded_matrix(n = 100_000, k = 2) = spdiagm([rand(n - abs(i)) for i = -k:k], -k:k)

function compare_storage(A, B)
    @show sizeof(B.nzval) / sizeof(A.nzval)
    @show sizeof(B.colptr) / sizeof(A.colptr)
    @show sizeof(B.rowval) / sizeof(A.rowval)

    A_size = sizeof(A.nzval) + sizeof(A.colptr) + sizeof(A.rowval)
    B_size = sizeof(B.nzval) + sizeof(B.colptr) + sizeof(B.rowval)

    @show B_size / A_size
end

"""
    example(n, k)

Benchmark A * x where A is SparseMatrixCSC vs CStyleBlockSparse where A is a banded
matrix with 2k + 1 diagonals and of order n. With k = 1 we get a very inefficient
CStyleBlockSparse, since roughly 50% of the stored values are zero. With k = 2 we only
store 16.67% zeros.
"""
function benchmark_banded(n = 100_000, k = 1)
    A = banded_matrix(n, k)
    B = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n)

    compare_storage(A, B)

    fst = @benchmark A_mul_B!(y, $B, $x) setup = (y = zeros($n))
    snd = @benchmark A_mul_B!(y, $A, $x) setup = (y = zeros($n))

    fst, snd
end

function benchmark_random(n = 100_000, k = 1)
    A = sprand(n, n, k / n)
    B = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n)

    compare_storage(A, B)

    fst = @benchmark A_mul_B!(y, $B, $x) setup = (y = zeros($n))
    snd = @benchmark A_mul_B!(y, $A, $x) setup = (y = zeros($n))

    fst, snd
end

function example(n = 100_000, k = 1)
    A = banded_matrix(n, 1)
    B = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n)

    A_mul_B!(zeros(n), B, x), A_mul_B!(zeros(n), A, x)
end

function benchmark2_banded(n = 100_000, k = 1)
    A = banded_matrix(n, k)
    B = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n, 2)

    compare_storage(A, B)

    fst = @benchmark A_mul_B!(y, $B, $x) setup = (y = zeros($n, 2))
    snd = @benchmark A_mul_B!(y, $A, $x) setup = (y = zeros($n, 2))

    fst, snd
end

function benchmark2_random(n = 100_000, k = 1)
    A = sprand(n, n, k / n)
    B = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n, 2)

    compare_storage(A, B)

    fst = @benchmark A_mul_B!(y, $B, $x) setup = (y = zeros($n, 2))
    snd = @benchmark A_mul_B!(y, $A, $x) setup = (y = zeros($n, 2))

    fst, snd
end

function example2(n = 100_000, k = 2)
    A = banded_matrix(n, k)
    B = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n, 2)

    A_mul_B!(zeros(n, 2), B, x), A_mul_B!(zeros(n, 2), A, x)
end

function example3(n = 100_000, k = 2)
    A = banded_matrix(n, k)
    B = convert(BlockSparseMatrixCSC{Float64,Int}, A)
    x = rand(n)

    A_mul_B!(zeros(n), B, x), A_mul_B!(zeros(n), A, x)
end

function bench_native(n = 100_000, k = 2)
    A = banded_matrix(n, k)
    B = convert(BlockSparseMatrixCSC{Float64,Int}, A)
    C = convert(CStyleBlockSparse{Float64,Int}, A)
    x = rand(n)

    bench_std    = @benchmark A_mul_B!(y, $A, $x) setup = (y = Vector{Float64}($n))
    bench_native = @benchmark A_mul_B!(y, $B, $x) setup = (y = Vector{Float64}($n))
    bench_c      = @benchmark A_mul_B!(y, $C, $x) setup = (y = Vector{Float64}($n))

    bench_std, bench_native, bench_c
end

function code_native()
    B = convert(BlockSparseMatrixCSC{Float64,Int}, banded_matrix(100, 2))
    x = reinterpret(BlockSparseMatrix.VecBlock{Float64}, rand(100))
    y = reinterpret(BlockSparseMatrix.VecBlock{Float64}, rand(100))
    
    @code_native A_mul_B!(y, B, x)
end
end