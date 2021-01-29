# scal!
scal!(a::Array{T}, p_size) where T = LinearAlgebra.BLAS.scal!(length(a), T(12/p_size), a, 1)
scal!(a::CuArray{T}, p_size) where T = CUBLAS.scal!(length(a), T(1/p_size), a)

# Matvec
dispgemm!(tA::Char, α::Float32, A::AbstractArray{Float32, 2}, b::AbstractArray{Float32, 1}, β::Float32, c::Array{Float32, 1}) = LinearAlgebra.BLAS.gemv!(tA, α, A, b, β, c)
dispgemm!(tA::Char, α::Float32, A::AbstractArray{Float32, 2}, b::AbstractArray{Float32, 1}, β::Float32, c::CuArray{Float32, 1}) = CUBLAS.gemv!(tA, α, A, b, β, c)

# MatMat
dispgemm!(tA::Char, tB::Char, α::Float32, A::AbstractArray{Float32, 2}, b::Array{Float32, 2}, β::Float32, c) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A, b, β, c)
dispgemm!(tA::Char, tB::Char, α::Float32, A::AbstractArray{Float32, 2}, b::CuArray{Float32, 2}, β::Float32, c) = CUBLAS.gemm!(tA, tB, α, A, b, β, c)

# Batched mat-mat
Bgemm!(tA, tB, α, A::AbstractArray, B::AbstractArray, β, C::Array) = NNlib.batched_gemm!(tA, tB, α, A, B, β, C)
Bgemm!(tA, tB, α, A::AbstractArray, B::AbstractArray, β, C::CuArray) = CUBLAS.gemm_strided_batched!(tA, tB, α, A, B, β, C)
