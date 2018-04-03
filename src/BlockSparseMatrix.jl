module BlockSparseMatrix

using StaticArrays

import Base: convert, full, A_mul_B!

export BlockSparseMatrixCSC, CStyleBlockSparse, native_A_mul_B!

const Block{Tv} = SMatrix{2,2,Tv,4}
const VecBlock{Tv} = SVector{2,Tv}
const BlockVector{Tv} = Vector{VecBlock{Tv}}

struct CStyleBlockSparse{Tv,Ti}
    m::Ti
    n::Ti
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Tv}
end

struct BlockSparseMatrixCSC{Tv,Ti}
    m::Ti
    n::Ti
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Block{Tv}}
end

# function full(A::BlockSparseMatrixCSC{Tv}) where {Tv}
#     F = zeros(2A.n, 2A.n)

#     nz_idx = 1

#     @inbounds for j = 1 : A.n
#         col = 2j - 1
#         for i = A.colptr[j] : A.colptr[j + 1] - 1
#             row = A.rowval[i]
#             F[row + 0, col + 0] = A.nzval[nz_idx + 0]
#             F[row + 1, col + 0] = A.nzval[nz_idx + 1]
#             F[row + 0, col + 1] = A.nzval[nz_idx + 2]
#             F[row + 1, col + 1] = A.nzval[nz_idx + 3]
#             nz_idx += 4
#         end
#     end

#     F
# end

"""
    to_block(A) -> (colptr, rowval, nzval)

Converts a SparseMatrixCSC to blocks. `colptr` is half the size, `rowval` keeps the
*original* row values and `nzval` is just a vector where elements start at indices 1:4:n.
"""
function to_block(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = size(A, 2)
    n_half = div(n, 2)
    
    # Optimistic sizehints
    nzval = sizehint!(Vector{Tv}(), div(length(A.nzval), 4))
    rowval = sizehint!(Vector{Ti}(), div(length(A.rowval), 2))
    colptr = Vector{Ti}(n_half + 1)
    colptr[1] = 1
    B = @MMatrix zeros(2, 2)
    
    # Loop over pairs of columns
    column = 0
    @inbounds for i = 1 : 2 : n
        j1 = A.colptr[i]
        j2 = A.colptr[i + 1]
        column += 1
        blocks = 0
        
        # As long as there's a value in either column i or i + 1
        # This can probably be simplified a bunch.
        while j1 < A.colptr[i + 1] || j2 < A.colptr[i + 2]
            blocks += 1

            fill!(B, zero(Tv))
            
            if j1 < A.colptr[i + 1] && j2 ≥ A.colptr[i + 2]
                row = A.rowval[j1]
            elseif j1 ≥ A.colptr[i + 1] && j2 < A.colptr[i + 2]
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
            
            push!(nzval, B...)
        end
        
        colptr[column + 1] = colptr[column] + blocks
    end

    colptr, rowval, nzval
end

# Rather inefficient way to convert SparseMatrixCSC to CStyleBlockSparse
function convert(::Type{CStyleBlockSparse{Tv,Ti}}, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    # Keep things simple
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("Matrix should be square"))
    iseven(size(A, 1)) || throw(DimensionMismatch("Matrix of even order"))

    colptr, rowval, nzval = to_block(A)
    n_half = div(A.n, 2)
    
    return CStyleBlockSparse(n_half, n_half, colptr, rowval, nzval)
end

function convert(::Type{BlockSparseMatrixCSC{Tv,Ti}}, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    # Keep things simple
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("Matrix should be square"))
    iseven(size(A, 1)) || throw(DimensionMismatch("Matrix of even order"))

    colptr, rowval, _nzval = to_block(A)
    nzval = reinterpret(Block{Tv}, _nzval)
    half_n = div(A.n, 2)

    # We'll index by element
    @inbounds for i = eachindex(rowval)
        rowval[i] = 1 + div(rowval[i] - 1, 2)
    end

    return BlockSparseMatrixCSC(half_n, half_n, colptr, rowval, nzval)
end

const bspmv = normpath(joinpath(@__DIR__, "bspmv.so"))
const bspmv2 = normpath(joinpath(@__DIR__, "bspmv2.so"))

function A_mul_B!(y::StridedVector{Float64}, A::CStyleBlockSparse{Float64,Int64}, x::StridedVector{Float64})
    fill!(y, 0.0)

    ccall((:bspmv, bspmv), 
          Void,
          (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          A.n, A.colptr, A.rowval, A.nzval, x, y)

    y
end

# Only implemented for x and y of size n x 2
function A_mul_B!(y::StridedMatrix{Float64}, A::CStyleBlockSparse{Float64,Int64}, x::StridedMatrix{Float64})
    size(y, 2) == size(x, 2) == 2 || throw(DimensionMismatch("Only n x 2 support atm."))

    fill!(y, 0.0)

    # void bspmv(int64_t n, int64_t * __restrict__ colptr, int64_t * __restrict__ rowval, double * __restrict__ nzval, double * __restrict__ x, double * __restrict__ y)
    ccall((:bspmv2, bspmv2), 
          Void,
          (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          A.n, A.colptr, A.rowval, A.nzval, x, y)

    y
end

function native_A_mul_B!(y::StridedVector{Tv}, A::CStyleBlockSparse{Tv,Ti}, x::StridedVector{Tv}) where {Tv,Ti}
    fill!(y, zero(Tv))

    nzidx = 1
    x_idx = 1
    A_idx = 1
    column = 1
    @inbounds while column ≤ A.n
        # Load the x values
        x_block = VecBlock{Tv}(x[x_idx], x[x_idx + 1])
        next_col = A.colptr[column + 1]
        while nzidx < next_col

            # Current row
            row = A.rowval[nzidx]
            y_block = VecBlock{Tv}(y[row], y[row + 1])

            # Load the next block
            A_block = Block{Tv}(
                A.nzval[A_idx + 0], 
                A.nzval[A_idx + 1], 
                A.nzval[A_idx + 2], 
                A.nzval[A_idx + 3]
            )

            # Do the computation
            tmp = y_block + A_block * x_block

            # Store
            y[row + 0] = tmp[1]
            y[row + 1] = tmp[2]

            # Increment counters
            A_idx += 4
            nzidx += 1
        end
        
        column += 1
        x_idx += 2
    end
    y
end
end