# Prevent scalars
CUDA.allowscalar(false)

"""
 Gradient via trce estimation
"""
function grad_ev(X::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4},
                 n::Integer, nw::Integer, stride::Integer=1)
    # Get right zero function
    ztype = typeof(X) <: CuArray ? CUDA.zeros : zeros
    nx, ny, nchi, batchsize = size(X)
    nxs, nys, ncho, batchsize = size(Y)
    dW = ztype(Float32, nw*nw, nchi, ncho)
    # Get diagonals offsets
    offsets = vcat([((-div(nw,2):div(nw,2)) .+ i*nx) for i=-div(nw,2):div(nw,2)]...)
    # Is there enough probing vectors to process them in batches (currentl min(n, 16))
    n > 1 ? be = div(n, 2^4)+1 : be = 1
    probe_bsize = min(2^4, n)
    # Normalize by number of probing vectors
    scale = 1/(be*probe_bsize)
    # LR product temporaries
    if be < n
        Re =  ztype(Float32, batchsize, probe_bsize)
        LRe = ztype(Float32, div(nx*ny*nchi, stride*stride), probe_bsize)
        LRees = ztype(Float32, nchi, ncho, probe_bsize)
        es = ztype(Float32, nx*ny, ncho, probe_bsize)
        e = ztype(Float32, div(nx*ny*ncho, stride*stride), probe_bsize)
    else
        Re = ztype(Float32, batchsize)
        LRe = ztype(Float32, div(nx*ny*nchi, stride*stride))
        es = ztype(Float32, nx*ny, ncho)
        e = ztype(Float32, div(nx*ny*ncho, stride*stride))
    end
    # reshape X and Y
    Xloc = reshape(X, :, batchsize)
    Yloc = reshape(Y, :, batchsize)
    # Probing
    for k=1:be
        be == n && LR_probe!(Xloc, Yloc, dW, Re, LRe, es, e, offsets, nx*ny)
        be < n && LR_probe_batched!(Xloc, Yloc, dW, Re, LRe, es, e, offsets, probe_bsize, nx*ny, LRees)
    end

    scal!(dW, scale)
    return reshape(dW, nw, nw, nchi, ncho)[end:-1:1, end:-1:1, :, :]
end


"""
LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 1}, LRe::AbstractArray{Float32, 1},
                   es::AbstractArray{Float32, 2}, e::AbstractArray{Float32, 2},
                   offsets::AbstractArray{<:Integer, 1}, nn::Integer)
"""
function LR_probe!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                   dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 1}, LRe::AbstractArray{Float32, 1},
                   es::AbstractArray{Float32, 2}, e::AbstractArray{Float32, 1},
                   offsets::AbstractArray{<:Integer, 1}, nn::Integer)
    # Probing vector
    disprand!(e)
    # TmpW
    tw = zeros(Float32, size(dW)[end-1:end]...)
    # R'*e
    dispgemv!('T', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemv!('N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    @inbounds for i=1:length(offsets)
        circshift!(es, reshape(e, nn, :), (offsets[i], 0))
        # Reshape as `nn x nci` and do Mat-Nat with es reshaped as nn*nco and accumulate
        dispgemm!('T', 'N', 1f0, reshape(LRe, nn, :), es, 0f0, tw)
        @views broadcast!(+, dW[i, :, :], dW[i, :, :], tw)
    end
end


"""
LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           es::AbstractArray{Float32, 3}, e::AbstractArray{Float32, 2},
                           offsets::AbstractArray{<:Integer, 1}, batch::Integer, nn::Integer,
                           LRees::AbstractArray{Float32, 3})
"""
function LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           es::AbstractArray{Float32, 3}, e::AbstractArray{Float32, 2},
                           offsets::AbstractArray{<:Integer, 1}, batch::Integer, nn::Integer,
                           LRees::AbstractArray{Float32, 3})
    # Probing vector
    disprand!(e)
    # R'*e
    dispgemm!('T', 'N', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemm!('N', 'N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    Lree = reshape(LRe, nn, :, batch)
    ee =  reshape(e, nn, :, batch)
    @inbounds for i=1:length(offsets)
        circshift!(es, ee, ((offsets[i]), 0, 0))
        # Reshape single probe as `nn x nci` and do Mat-vec with es dW for all input channels and sum
        Bgemm!(R)('T', 'N', 1f0, Lree, es, 0f0, LRees)
        cumsum!(LRees, LRees, dims=3)
        @views broadcast!(+, dW[i, :, :], dW[i, :, :], LRees[:, :, batch])
    end
end

# rand
for N=1:3
    @eval function disprand!(e::CuArray{Float32, $N})
        CUDA.rand!(e)
        broadcast!(-, e, e, .5f0)
    end
    @eval function disprand!(e::Array{Float32, $N})
        rand!(e)
        broadcast!(-, e, e, .5f0)
    end
end
