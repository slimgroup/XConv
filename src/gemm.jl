
import CUDA: unsafe_free!
# Matvec
dispgemv!(tA::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 1}, β::Float32, c::Array{Float32, 1}) = LinearAlgebra.BLAS.gemv!(tA, α, A, b, β, c)
dispgemv!(tA::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 1}, β::Float32, c::CuArray{Float32, 1}) = CUBLAS.gemv!(tA, α, A, b, β, c)

# MatMat
dispgemm!(tA::Char, tB::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 2}, β::Float32, c::Array{Float32, 2}) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A, b, β, c)
dispgemm!(tA::Char, tB::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 2}, β::Float32, c::CuArray{Float32, 2}) = CUBLAS.gemm!(tA, tB, α, A, b, β, c)

# Batched mat-mat
Bgemm!(::Array) = NNlib.batched_gemm!
Bgemm!(::CuArray) = gemm_batched!

# CUDA batched gemm since theirs only work on vector of matrix, not on 3D matrices
# create a batch of pointers in device memory from a batch of device arrays
@inline function unsafe_batch(batch::CuArray{Float32, 3}) where {T}
    ptrb = pointer(batch)
    strb = Base.stride(batch, 3)
    ptrs = CuArray([ptrb + (k-1) * strb * sizeof(Float32) for k=1:size(batch, 3)])
    return ptrs
end

function gemm_batched!(transA::Char,  transB::Char, alpha::Number,
                       A::CuArray{Float32, 3}, B::CuArray{Float32, 3},
                       beta::Number, C::CuArray{Float32, 3})

    if size(A, 3) != size(B, 3) || size(A, 3) != size(C, 3)
        throw(DimensionMismatch(""))
    end

    for b=1:size(A, 3)
        m = size(A, transA == 'N' ? 1 : 2)
        k = size(B, transA == 'N' ? 2 : 1)
        n = size(C, transB == 'N' ? 2 : 1)
        if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
            throw(DimensionMismatch(""))
        end
    end

    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)

    lda = max(1,stride(A, 2))
    ldb = max(1,stride(B, 2))
    ldc = max(1,stride(C, 2))

    Aptrs = unsafe_batch(A)
    Bptrs = unsafe_batch(B)
    Cptrs = unsafe_batch(C)

    CUBLAS.cublasSgemmBatched(CUBLAS.handle(), transA, transB, m, n, k, alpha, Aptrs, lda, Bptrs,
                              ldb, beta, Cptrs, ldc, size(A, 3))
    unsafe_free!(Cptrs)
    unsafe_free!(Bptrs)
    unsafe_free!(Aptrs)
end
