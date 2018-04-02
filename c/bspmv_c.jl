using BenchmarkTools

import Base: convert

struct BlockSparseMatrixCSC{Tv,Ti}
    n::Ti
    nzval::Vector{Tv}
    rowval::Vector{Ti}
    colptr::Vector{Ti}
end

function convert(::Type{BlockSparseMatrixCSC{Tv,Ti}}, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = size(A, 2)

    # Keep things simple
    @assert size(A, 1) == size(A, 2)
    @assert iseven(n)
    
    nzval = Vector{Tv}()
    rowval = Vector{Ti}()
    colptr = Vector{Ti}(div(n, 2) + 1)
    colptr[1] = 1
    
    # Loop over pairs of columns
    column = 0
    for i = 1 : 2 : n
        j1 = A.colptr[i]
        j2 = A.colptr[i + 1]
        column += 1
        blocks = 0
        
        while j1 < A.colptr[i + 1] || j2 < A.colptr[i + 2]
            B = zeros(2,2)
            blocks += 1
            
            if j1 < A.colptr[i + 1] && j2 ≥ A.colptr[i + 2]
                row = A.rowval[j1]
            elseif j2 ≥ A.colptr[i + 1] && j1 < A.colptr[i + 2]
                row = A.rowval[j2]
            else
                row = min(A.rowval[j1], A.rowval[j2])
            end
            
            # Make sure we start at an odd row value
            if row % 2 == 0
                row -= 1
            end
            
            push!(rowval, row)
            
            # Column 1
            if j1 < A.colptr[i + 1] && A.rowval[j1] == row
                B[1,1] = A.nzval[j1]
                j1 += 1
            end
            
            if j1 < A.colptr[i + 1] && A.rowval[j1] == row + 1
                B[2,1] = A.nzval[j1]
                j1 += 1
            end
            
            # Column 2
            if j2 < A.colptr[i + 2] && A.rowval[j2] == row
                B[1,2] = A.nzval[j2]
                j2 += 1
            end
            
            if j2 < A.colptr[i + 2] && A.rowval[j2] == row + 1
                B[2,2] = A.nzval[j2]
                j2 += 1
            end
            
            push!(nzval, B[:]...)
        end
        
        colptr[column + 1] = colptr[column] + blocks
    end
    
    return BlockSparseMatrixCSC(div(n, 2), nzval, rowval, colptr)
end

banded_matrix(n = 100_000, k = 2) = spdiagm([rand(n - abs(i)) for i = -k:k], -k:k)

function mul!(y::StridedVector{Float64}, A::BlockSparseMatrixCSC{Float64,Int64}, x::StridedVector{Float64})
    # void bspmv(int64_t n, int64_t * __restrict__ colptr, int64_t * __restrict__ rowval, double * __restrict__ nzval, double * __restrict__ x, double * __restrict__ y)
    ccall((:bspmv, "./bspmv.so"), 
          Void,
          (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          A.n, A.colptr, A.rowval, A.nzval, x, y)

    y
end

"""
    example(n, k)

Benchmark A * x where A is SparseMatrixCSC vs BlockSparseMatrixCSC where A is a banded
matrix with 2k + 1 diagonals and of order n. With k = 1 we get a very inefficient
BlockSparseMatrixCSC, since roughly 50% of the stored values are zero. With k = 2 we only
store 16.67% zeros.
"""
function benchmark(n = 100_000, k = 1)
    A = banded_matrix(n, 1)
    B = convert(BlockSparseMatrixCSC{Float64,Int}, A)
    x = rand(n)

    fst = @benchmark mul!(y, $B, $x) setup = (y = zeros($n))
    snd = @benchmark A_mul_B!(y, $A, $x) setup = (y = zeros($n))

    fst, snd
end

function example(n = 100_000, k = 1)
    A = banded_matrix(n, 1)
    B = convert(BlockSparseMatrixCSC{Float64,Int}, A)
    x = rand(n)

    mul!(zeros(n), B, x), A * x
end