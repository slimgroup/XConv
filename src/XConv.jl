module XConv

# Dependencies
using Random
using LinearAlgebra, IterTools
using CUDA, GPUArrays
using NNlib
using ChainRulesCore

import NNlib: colmajor

export grad_ev, initXConv

# Setup valuesa
const DEFAULT = "TrueGrad"
const EV = "EVGrad"
const _params = Dict(:p_size => 2^4, :mode => DEFAULT)

# Probing
include("gemm.jl")
include("probe.jl")

# EV setup function
function initXConv(probe_size::Integer, mode::String)
    _params[:mode] = mode
    _params[:p_size] = probe_size
end

# rrule
function ChainRulesCore.rrule(::typeof(NNlib.conv), X, w, cdim::DenseConvDims; kw...)
    Y = conv(X, w, cdim)
    back = _params[:mode] == EV ? Δconv_ev(X, w, cdim; kw...) : Δconv_std(X, w, cdim; kw...)
    return Y, back
end

function Δconv_std(x, w, cdim; kw...)
    function back(Δ)
        Δ = colmajor(Δ)
        return (
            NO_FIELDS,
            @thunk(∇conv_data(Δ, w, cdim, kw...)),
            @thunk(∇conv_filter(x, Δ, cdim, kw...)),
            NoTangent(),
        )
    end
    return back
end

function Δconv_ev(X, w, cdim; kw...)
    seed, eX = probe_X(X)
    function back(Δ)
        Δ = colmajor(Δ)
        return (
            NO_FIELDS,
            @thunk(∇conv_data(Δ, w, cdim, kw...)),
            @thunk(grad_ev(seed, eX, Δ, w, NNlib.stride(cdim))),
            NoTangent(),
        )
    end
    return back
end

end
