module XConv

# Dependencies
using Random
using LinearAlgebra
using CUDA
using NNlib
using ChainRulesCore

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

# Helper functions for rrule
weightsize(::DenseConvDims{N,K,C_in,C_out,S,P,D,F}) where {N,K,C_in,C_out,S,P,D,F} = K[1]

colmajor(x) = (NNlib.is_strided(x) && Base.stride(x, 1) == 1) ? x : collect(x)

# Little wrapper to have our own conv
for N=3:5
    for AT in [Array, CuArray]
        @eval begin
            function NNlib.∇conv_filter(x::$(AT){xT, $N}, dy::$(AT){wT, $N},
                                        cdims::ConvDims; kw...) where {xT, wT}
                dy = colmajor(dy)
                _params[:mode] == EV && return grad_ev(x, dy, _params[:p_size], weightsize(cdims))
                return invoke(NNlib.∇conv_filter,  Tuple{AbstractArray{Float32, 4}, AbstractArray{Float32, 4}, ConvDims},
                              x, dy, cdims; kw...)
            end
        end
    end
end

end
