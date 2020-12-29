module XConv

using LinearAlgebra, ChainRulesCore, CUDA

export grad_ev

# CPU version
include("probe.jl")

# Main routine
function grad_ev(X::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4},
                 n::Integer, nw::Integer, stride::Integer=1)
    # Get right zero function
    ztype = typeof(X) <: CuArray ? CUDA.zeros : zeros
    nx, ny, nchi, batchsize = size(X)
    nxs, nys, ncho, batchsize = size(Y)
    dW = ztype(Float32, nw*nw, nchi, ncho)
    # Get diagonals offsets
    offsets = vcat([((-1:1) .+ i*nx) for i=-div(nw,2):div(nw,2)]...)
    # subsample?
    scale = n*(nx+ny)*sqrt(batchsize)
    # Is there enough probing vectors to process them in batches (currentl min(n, 16))
    n > 1 ? be = div(n, 2^4)+1 : be = 1
    probe_bsize = min(2^4, n)
    # LR product temporaries
    if be < n
        Re =  ztype(Float32, batchsize, probe_bsize)
        LRe = ztype(Float32, div(nx*ny*nchi, stride*stride), probe_bsize)
        es = ztype(Float32, nx*ny, probe_bsize)
    else
        Re = ztype(Float32, batchsize)
        LRe = ztype(Float32, div(nx*ny*nchi, stride*stride))
        es = ztype(Float32, nx*ny)
    end
    # reshape X and Y
    Xloc = reshape(X, :, batchsize)
    Yloc = reshape(Y, :, batchsize)
    # Probing
    for k=1:be
        be == n && LR_probe!(Xloc, Yloc, dW, Re, LRe, es, offsets, nx*ny)
        be < n && LR_probe_batched!(Xloc, Yloc, dW, Re, LRe, es, offsets, probe_bsize, nx*ny)
    end

    rdiv!(dW, scale)
    return reshape(dW, nw, nw, nchi, ncho)[end:-1:1, end:-1:1, :, :]
end

end