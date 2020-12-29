"""
LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::AbstractArray{Float32, 4}, Re::AbstractArray{Float32, 1}, LRe::AbstractArray{Float32, 1},
                   offsets::AbstractArray{<:Integer, 1})
"""
function LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 1}, LRe::AbstractArray{Float32, 1},
                   es::AbstractArray{Float32, 1}, offsets::AbstractArray{<:Integer, 1}, nn::Integer)
    # Temps
    e = disprand(R)
    # R'*e
    dispgemv!('T', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemv!('N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        @inbounds for co=1:size(dW)[end]
            circshift!(es, view(e, (co-1)*nn+1:co*nn), offsets[i])
            # Reshape as `nn x nci` and do Mat-vec with es dW for all input channels and sum
            dispgemv!('T', 1f0, reshape(LRe, nn, :), es, 1f0, view(dW, i, :, co))
        end
    end
end


"""
LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 4}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           offsets::AbstractArray{<:Integer, 1}, batch::Integer)
"""
function LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           es::AbstractArray{Float32, 2}, offsets::AbstractArray{<:Integer, 1},
                           batch::Integer, nn::Integer)
    # Temps
    e = disprand(R, batch)
    # R'*e
    dispgemm!('T', 'N', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemm!('N', 'N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        @inbounds for co=1:size(dW)[end]
            st = (co-1)*nn
            circshift!(es, view(e, st+1:st+nn, :), (offsets[i], 0))
            @inbounds for b=1:batch
                # Reshape single probe as `nn x nci` and do Mat-vec with es dW for all input channels and sum
                mul!(view(dW, i, :, co), reshape(view(LRe, :, b), nn, :)', view(es, :, b), 1f0, 1f0)
            end
        end
    end
end

# rand
disprand(R::CuArray) = CUDA.rand([-1f0, 1f0], size(R, 1))
disprand(R::Array) = rand([-1f0, 1f0], size(R, 1))
disprand(R::CuArray, b::Integer) = CUDA.rand([-1f0, 1f0], size(R, 1), b)
disprand(R::Array, b::Integer) = rand([-1f0, 1f0], size(R, 1), b)

# Matvec
dispgemv!(tA::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 1}, β::Float32, c::Array{Float32, 1}) = LinearAlgebra.BLAS.gemv!(tA, α, A, b, β, c)
dispgemv!(tA::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 1}, β::Float32, c::CuArray{Float32, 1}) = CUBLAS.gemv!(tA, α, A, b, β, c)

# MatMat
dispgemm!(tA::Char, tB::Char, α::Float32, A::Array{Float32, 2}, b::Array{Float32, 2}, β::Float32, c::Array{Float32, 2}) = LinearAlgebra.BLAS.gemm!(tA, tB, α, A, b, β, c)
dispgemm!(tA::Char, tB::Char, α::Float32, A::CuArray{Float32, 2}, b::CuArray{Float32, 2}, β::Float32, c::CuArray{Float32, 2}) = CUBLAS.gemm!(tA, tB, α, A, b, β, c)
