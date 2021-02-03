# Prevent scalars
CUDA.allowscalar(false)

"""
 Gradient via trce estimation
"""
function grad_ev(X::AbstractArray{Float32, 4}, Y::AbstractArray{Float32, 4},
                 n::Integer, nw::Integer, stride::NTuple=(1, 1))

    nx, ny, nchi, batchsize = size(X)
    nxs, nys, ncho, batchsize = size(Y)
    dW = similar(X, nw*nw, nchi, ncho)
    fill!(dW, 0f0)

    # Get diagonals offsets
    offsets = vcat([((-div(nw,2):div(nw,2)) .+ i*nx) for i=-div(nw,2):div(nw,2)]...)

    # LR product temporaries
    Re =  similar(X, batchsize, n)
    LRe = similar(X, nx*ny*nchi, n)
    LRees = similar(X, nchi, ncho, n)
    e = similar(X, nx*ny*ncho, n)
    disprand!(e)

    # reshape X and Y
    Xloc = reshape(X, :, batchsize)
    Yloc = reshape(Y, :, batchsize)

    # Probing
    LR_probe_batched!(Xloc, Yloc, dW, Re, LRe, e, offsets, n, (nx,ny), LRees, stride)

    scal!(dW, n)
    return reshape(dW, nw, nw, nchi, ncho)[end:-1:1, end:-1:1, :, :]
end


function grad_ev(seeding::Union{UInt64, Tuple{GPUArrays.RNG,UInt64}}, eX::AbstractArray{Float32, 2},
		         Y::AbstractArray{Float32, 4}, w::AbstractArray{Float32}, stride::NTuple=(1, 1))
    n = _params[:p_size]
    nw, _, nchi, ncho = size(w)
    nx, ny = size(Y)[1:2]
    batchsize, probe_bsize = size(eX)
    dW = similar(w, nw*nw, nchi, ncho)
    fill!(dW, 0f0)

    # Get diagonals offsets
    offsets = vcat([((-div(nw,2):div(nw,2)) .+ i*nx*stride[1]) for i=-div(nw,2):div(nw,2)]...)

    # LR product temporaries
    LRe = similar(Y, nx*ny*ncho*prod(stride), probe_bsize)
    LRees = similar(Y, nchi, ncho, probe_bsize)
    e = similar(Y, nx*ny*nchi*prod(stride), probe_bsize)
    disprand!(e, seeding)
    
    # reshape X and Y
    Yloc = dilate(Y, nx*stride[1], stride)

    # Probing
    LR_probe_batched!(Yloc, eX, dW, LRe, e, offsets, probe_bsize, (nx, ny), LRees, stride)

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
                           batch::Integer, N::NTuple, LRees::AbstractArray{Float32, 3},
                           stride::NTuple=(1, 1))
    se = maximum(offsets)
    nn = prod(N)
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
                           batch::Integer, N::NTuple, LRees::AbstractArray{Float32, 3},
                           stride::NTuple=(1, 1))
    se = maximum(offsets)
    nn = prod(N)
    eend = nn-se
    # Apply stride
    e, e_s = strided_e(e, N..., stride)
    # R'*e
    dispgemm!('T', 'N', 1f0, R, e_s, 0f0, Re)
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

function probe_X(X::AbstractArray{Float32, 4}, e::AbstractArray{Float32})
    eX = similar(X, size(X, 4), _params[:p_size])
    dispgemm!('T', 'N', 1f0, reshape(X, :,  size(X, 4)), e, 0f0, eX)
    return eX
end

function probe_vec(seeding, X::AbstractArray{Float32, 4})
    e = similar(X, prod(size(X)[1:3]), _params[:p_size])
    disprand!(e, seeding)
    return e
end

@inline make_rng(::CuArray) = (GPUArrays.default_rng(CuArray), rand(UInt64))
@inline make_rng(::Array) = rand(UInt64)

@inline function strided_e(e::AbstractArray{Float32, 2}, nx::Integer, ny::Integer, stride::NTuple=(1, 1))
    stride == (1, 1) && (return e, e)
    inds = vec(sum.(product(0:stride[1]:nx-1, 1:nx*stride[2]:size(e, 1))))
    e_s = e[inds, :]
    return e, e_s
end

@inline function dilate(Y::AbstractArray{Float32, 4}, nx::Integer, stride::NTuple=(1, 1))
    Y = reshape(Y, :, size(Y, 4))
    stride == (1, 1) && (return Y)
    Yloc = similar(Y, size(Y, 1)*prod(stride), size(Y, 2))
    fill!(Yloc, 0f0)
    inds = vec(sum.(product(0:stride[1]:nx-1, 1:nx*stride[2]:size(Yloc, 1))))
    Yloc[inds, :] .= Y[:, :]
    return Yloc
end