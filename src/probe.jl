# Prevent scalars
CUDA.allowscalar(false)

"""
 Gradient via trce estimation
"""
function grad_ev(X::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4},
                 n::Integer, nw::Integer, stride::Integer=1)

    nx, ny, nchi, batchsize = size(X)
    nxs, nys, ncho, batchsize = size(Y)
    dW = similar(X, nw*nw, nchi, ncho)
    fill!(dW, 0f0)

    # Get diagonals offsets
    offsets = vcat([((-div(nw,2):div(nw,2)) .+ i*nx) for i=-div(nw,2):div(nw,2)]...)

    # Is there enough probing vectors to process them in batches (currentl min(n, 16))
    be = div(n, 2^4)+1
    probe_bsize = min(2^4, n)

    # LR product temporaries
    Re =  similar(X, batchsize, probe_bsize)
    LRe = similar(X, div(nx*ny*nchi, stride*stride), probe_bsize)
    LRees = similar(X, nchi, ncho, probe_bsize)
    e = similar(X, div(nx*ny*ncho, stride*stride), probe_bsize)

    # reshape X and Y
    Xloc = reshape(X, :, batchsize)
    Yloc = reshape(Y, :, batchsize)

    # Probing
    for k=1:be
        LR_probe_batched!(Xloc, Yloc, dW, Re, LRe, e, offsets, probe_bsize, nx*ny, LRees)
    end
    scal!(dW, probe_bsize)
    return reshape(dW, nw, nw, nchi, ncho)[end:-1:1, end:-1:1, :, :]
end


function grad_ev(seeding::Union{UInt64, Tuple{GPUArrays.RNG,UInt64}}, eX::AbstractArray{Float32, 2},
		 Y::AbstractArray{Float32, 4},
                 w::AbstractArray{Float32}, stride::Integer=1)
    n = _params[:p_size]
    nw, _, nchi, ncho = size(w)
    nx, ny = size(Y)[1:2]
    batchsize, probe_bsize = size(eX)
    dW = similar(w, nw*nw, nchi, ncho)
    fill!(dW, 0f0)

    # Get diagonals offsets
    offsets = vcat([((-div(nw,2):div(nw,2)) .+ i*nx) for i=-div(nw,2):div(nw,2)]...)

    # LR product temporaries
    LRe = similar(Y, div(nx*ny*ncho, stride*stride), probe_bsize)
    LRees = similar(Y, nchi, ncho, probe_bsize)
    e = similar(Y, div(nx*ny*nchi, stride*stride), probe_bsize)
    disprand!(e, seeding)
    
    # reshape X and Y
    Yloc = reshape(Y, :, batchsize)

    # Probing
    LR_probe_batched!(Yloc, eX, dW, LRe, e, offsets, probe_bsize, nx*ny, LRees)

    scal!(dW, probe_bsize)
    any(isnan.(dW)) && scal!(disprand!(dW), 1f3)
    return reshape(dW, nw, nw, nchi, ncho)[end:-1:1, end:-1:1, :, :]
end

"""
LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           es::AbstractArray{Float32, 3}, e::AbstractArray{Float32, 2},
                           offsets::AbstractArray{<:Integer, 1}, batch::Integer, nn::Integer,
                           LRees::AbstractArray{Float32, 3})
"""
function LR_probe_batched!(L::AbstractArray{Float32, 2}, Re::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, LRe::AbstractArray{Float32, 2},
                           e::AbstractArray{Float32, 2}, offsets::AbstractArray{<:Integer, 1},
                           batch::Integer, nn::Integer, LRees::AbstractArray{Float32, 3})
    se = maximum(offsets)
    eend = nn-se
    # L*R'*e
    dispgemm!('N', 'N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    Lree = view(reshape(LRe, nn, :, batch), se+1:eend, :, :)
    ee = reshape(e, nn, :, batch)
    @inbounds for i=1:length(offsets)
        # Reshape single probe as `nn x nci` and do Mat-vec with es dW for all input channels and sum
        ev = view(ee, se+1+offsets[i]:eend+offsets[i], :, :)
        Bgemm!('T', 'N', 1f0, ev, Lree, 0f0, LRees)
        cumsum!(LRees, LRees, dims=3)
        @views broadcast!(+, dW[i, :, :], dW[i, :, :], LRees[:, :, batch])
    end
end

function LR_probe_batched!(L::AbstractArray{Float32, 2}, R::AbstractArray{Float32, 2},
                           dW::AbstractArray{Float32, 3}, Re::AbstractArray{Float32, 2}, LRe::AbstractArray{Float32, 2},
                           e::AbstractArray{Float32, 2}, offsets::AbstractArray{<:Integer, 1},
                           batch::Integer, nn::Integer, LRees::AbstractArray{Float32, 3})
    se = maximum(offsets)
    eend = nn-se
    # Probing vector
    disprand!(e)
    # R'*e
    dispgemm!('T', 'N', 1f0, R, e, 0f0, Re)
    # L*R'*e
    dispgemm!('N', 'N', 1f0, L, Re, 0f0, LRe)
    # e'*L*R'*e
    Lree = view(reshape(LRe, nn, :, batch), se+1:eend, :, :)
    ee = reshape(e, nn, :, batch)
    @inbounds for i=1:length(offsets)
        # Reshape single probe as `nn x nci` and do Mat-vec with es dW for all input channels and sum
        ev = view(ee, se+1-offsets[i]:eend-offsets[i], :, :)
        Bgemm!('T', 'N', 1f0, Lree, ev, 0f0, LRees)
        cumsum!(LRees, LRees, dims=3)
        @views broadcast!(+, dW[i, :, :], dW[i, :, :], LRees[:, :, batch])
    end
end

# rand
disprand!(e::Array{Float32}, seed::UInt64) = (Random.seed!(seed);disprand!(e))
disprand!(e::AbstractArray{Float32}) = (rand!(e);broadcast!(-, e, e, .5f0))
disprand!(e::CuArray{Float32}, seeding::Tuple{GPUArrays.RNG, UInt64}) = (Random.seed!(seeding...);randn!(seeding[1], e);)

# Utilities
function probe_X(X::AbstractArray{Float32, 4})
    seeding = make_rng(X)
    e = probe_vec(seeding, X)
    eX = similar(X, size(X, 4), _params[:p_size])
    dispgemm!('T', 'N', 1f0, reshape(X, :,  size(X, 4)), e, 0f0, eX)
    return seeding, eX
end

function probe_vec(seeding, X::AbstractArray{Float32, 4})
    e = similar(X, prod(size(X)[1:3]), _params[:p_size])
    disprand!(e, seeding)
    return e
end

@inline make_rng(::CuArray) = (GPUArrays.default_rng(CuArray), rand(UInt64))
@inline make_rng(::Array) = rand(UInt64)
