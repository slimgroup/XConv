"""
LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::AbstractArray{Float32, 4}, Re::AbstractArray{Float32, 1}, LRe::AbstractArray{Float32, 1},
                   offsets::AbstractArray{<:Integer, 1})
"""
function LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 1}, LRe::AbstractArray{Float32, 1},
                   es::AbstractArray{Float32, 1}, offsets::AbstractArray{<:Integer, 1}, nn::Integer)
    # Probing vector
    e = disprand(R)
    # R'*e
    dispgemv!('T', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemv!('N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        circshift!(es, reshape(e, nn, :), (offsets[i], 0))
        # Reshape as `nn x nci` and do Mat-Nat with es reshaped as nn*nco and accumulate
        dispgemm!('T', 'N', 1f0, reshape(LRe, nn, :), es, 1f0, view(dW, i, :, :))
    end
end


"""
LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 4}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           offsets::AbstractArray{<:Integer, 1}, batch::Integer)
"""
function LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           es::AbstractArray{Float32, 3}, offsets::AbstractArray{<:Integer, 1},
                           batch::Integer, nn::Integer, LRees::Array{Float32, 3})
    # Probing vector
    e = disprand(R, batch)
    # R'*e
    dispgemm!('T', 'N', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemm!('N', 'N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        circshift!(es, reshape(e, nn, :, batch), (offsets[i], 0, 0))
        # Reshape single probe as `nn x nci` and do Mat-vec with es dW for all input channels and sum
        Bgemm!(R)('T', 'N', 1f0, reshape(LRe, nn, :, batch), es, 0f0, LRees)
        cumsum!(LRees, LRees, dims=3)
        @views broadcast!(+, dW[i, :, :], dW[i, :, :], LRees[:, :, batch])
    end
end

# rand
disprand(R::CuArray) = cu(rand([-1f0, 1f0], size(R, 1)))
disprand(R::Array) = rand([-1f0, 1f0], size(R, 1))
disprand(R::CuArray, b::Integer) = cu(rand([-1f0, 1f0], size(R, 1), b))
disprand(R::Array, b::Integer) = rand([-1f0, 1f0], size(R, 1), b)

# Matvec
dispgemv!(tA::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 1}, β::Float32, c::Array{Float32, 1}) = LinearAlgebra.BLAS.gemv!(tA, α, A, b, β, c)
dispgemv!(tA::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 1}, β::Float32, c::CuArray{Float32, 1}) = CUBLAS.gemv!(tA, α, A, b, β, c)

# MatMat
dispgemm!(tA::Char, tB::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 2}, β::Float32, c::Array{Float32, 2}) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A, b, β, c)
dispgemm!(tA::Char, tB::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 2}, β::Float32, c::CuArray{Float32, 2}) = CUBLAS.gemm!(tA, tB, α, A, b, β, c)

# Batched mat-mat
Bgemm!(::Array) = NNlib.batched_gemm!
Bgemm!(::CuArray) = CUBLAS.batched_gemm!