module XConv

using LinearAlgebra
using ChainRules
using CUDA
using NNlib
import NNlib: is_strided

CUDA.allowscalar(false)

export grad_ev, initXConv

# Setup valuesa
const DEFAULT = "TrueGrad"
const EV = "EVGrad"
const _params = Dict(:p_size => 2^4, :mode => EV)

function initXConv(probe_size=2^4, mode=EV)
    _params[:mode] = mode
    _params[:p_size] = probe_size
end

# Probing
include("gemm.jl")
include("probe.jl")
# Redefine rrule for conv

function ChainRules.rrule(::typeof(NNlib.conv_im2col), x, w, cdims; kw...)
    function conv_pullback(Δ)
        ∇conv_filter_th = _params[:mode] == EV ? @thunk(grad_ev(x, Δ, _params[:p_size], size(w, 1))) : @thunk(NNlib.∇conv_filter(x, Δ, cdims, kw...))
        return (
            NO_FIELDS,
            @thunk(NNlib.∇conv_data(Δ, w, cdims, kw...)),
            ∇conv_filter_th,
            DoesNotExist(),
        )
    end
    return NNlib.conv(x, w, cdims; kw...), conv_pullback
end


end