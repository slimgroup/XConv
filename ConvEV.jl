module EVC

using LinearAlgebra, ChainRulesCore
import LinearAlgebra.BLAS: gemv!, gemm!, dot, scal!


export grad_ev


"""
    grad_ev(X::Array{Float32, 4}, Y::Array{Float32, 4},
            n::Integer, nw::Integer, stride::Integer=1)
"""
function grad_ev(X::Array{Float32, 4}, Y::Array{Float32, 4},
                 n::Integer, nw::Integer, stride::Integer=1)

    nx, ny, nchi, batchsize = size(X)
    nxs, nys, ncho, batchsize = size(Y)
    dW = zeros(Float32, nw*nw, nchi, ncho)
    # Get diagonals offsets
    offsets = vcat([((-1:1) .+ i*nx) for i=-div(nw,2):div(nw,2)]...)
    # subsample?
    scale = n*(nx+ny)*sqrt(batchsize)
    # Is there enough probing vectors to process them in batches (currently of 16)
    n > 2^4 ? be = div(n, 2^4)+1 : be = n
    probe_bsize = min(2^4, n)
    # LR product temporaries
    Re = be < n ? zeros(Float32, batchsize, probe_bsize) : zeros(Float32, batchsize)
    LRe = be < n ? zeros(Float32, div(nx*ny*nchi, stride*stride), probe_bsize) : zeros(Float32, div(nx*ny*nchi, stride*stride))

    # reshape X and Y
    Xloc = reshape(X, :, batchsize)
    Yloc = reshape(Y, :, batchsize)
    # Probing
    for k=1:be
        be == n && LR_probe!(Xloc, Yloc, dW, Re, LRe, offsets, nx*ny)
        be < n && LR_probe_batched!(Xloc, Yloc, dW, Re, LRe, offsets, probe_bsize, nx*ny)
    end

    rdiv!(dW, scale)
    return reshape(dW, nw, nw, nchi, ncho)[end:-1:1, end:-1:1, :, :]
end


"""
LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::Array{Float32, 4}, Re::Array{Float32, 1}, LRe::Array{Float32, 1},
                   offsets::Array{<:Integer, 1})
"""
function LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::Array{Float32, 3}, Re::Array{Float32, 1}, LRe::Array{Float32, 1},
                   offsets::Array{<:Integer, 1}, nn::Integer)
    # Temps
    e = rand([-1f0, 1f0], size(R, 1))
    es = zeros(Float32, nn)
    # R'*e
    gemv!('T', 1f0, R, e, 0f0, Re)
    # L*R'*e
    gemv!('N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        @inbounds for co=1:size(dW)[end]
            st = (co-1)*nn
            circshift!(es, view(e, st+1:st+nn), offsets[i])
            @inbounds for ci=1:size(dW)[end-1]
                st2 = (ci-1)*nn
                dW[i, ci, co] += dot(nn, es, 1, view(LRe, st2+1:st2+nn), 1)
            end
        end
    end
end


"""
LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::Array{Float32, 4}, Re::Array{Float32, 2}, LRe::Array{Float32, 2},
                           offsets::Array{<:Integer, 1}, batch::Integer)
"""
function LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::Array{Float32, 3}, Re::Array{Float32, 2}, LRe::Array{Float32, 2},
                           offsets::Array{<:Integer, 1}, batch::Integer, nn::Integer)
    # Temps
    e = rand([-1f0, 1f0], size(R, 1), batch)
    es = zeros(Float32, nn, batch)
    # R'*e
    gemm!('T', 'N', 1f0, R, e, 0f0, Re)
    # L*R'*e
    gemm!('N', 'N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        @inbounds for co=1:size(dW)[end]
            st = (co-1)*nn
            circshift!(es, view(e, st+1:st+nn, :), (offsets[i], 0))
            @inbounds for ci=1:size(dW)[end-1]
                st2 = (ci-1)*nn
                @inbounds for b=1:batch
                    dW[i, ci, co] += dot(nn, view(es, :, b), 1, view(LRe, st2+1:st2+nn, b), 1)
                end
            end
        end
    end
end


end