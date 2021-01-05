# scal!
scal!(a::Array{T}, s) where T = LinearAlgebra.BLAS.scal!(length(a), T(s), a, 1)
scal!(a::CuArray{T}, s) where T = CUBLAS.scal!(length(a), T(s), a)

# Matvec
dispgemv!(tA::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 1}, β::Float32, c::Array{Float32, 1}) = LinearAlgebra.BLAS.gemv!(tA, α, A, b, β, c)
dispgemv!(tA::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 1}, β::Float32, c::CuArray{Float32, 1}) = CUBLAS.gemv!(tA, α, A, b, β, c)

# MatMat
dispgemm!(tA::Char, tB::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 2}, β::Float32, c) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A, b, β, c)
dispgemm!(tA::Char, tB::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 2}, β::Float32, c) = CUBLAS.gemm!(tA, tB, α, A, b, β, c)

# Batched mat-mat
Bgemm!(::Array) = NNlib.batched_gemm!
Bgemm!(::CuArray) = CUBLAS.gemm_strided_batched!

