module XConv

using LinearAlgebra
using ChainRules
using CUDA
using NNlib
import NNlib: is_strided

CUDA.allowscalar(false)

export grad_ev, initXConv

# Setup values
const DEFAULT = "TrueGrad"
const EV = "EVGrad"
const _probe_size = 2^4
const _mode = EV

function initXConv(probe_size=2^4, mode=EV)
    _mode = mode
    _probe_size = probe_size
end

# Probing
include("gemm.jl")
include("probe.jl")
# Redefine rrule for conv
colmajor(x) = (is_strided(x) && Base.stride(x, 1) == 1) ? x : collect(x)
NNlib.conv(x, w, cdims) = myconv(x, w, cdims) 
myconv(x, w, cdims) = invoke(NNlib.conv, Tuple{Any, Any, Any}, x, w, cdims)


function ChainRules.rrule(::typeof(myconv), x, w, cdims; kw...)
    function conv_pullback(Δ)
        println("hello ", cdims, ", ", EV)
        Δ = colmajor(Δ)
        ∇conv_filter_th = _mode == EV ?  @thunk(grad_ev(x, Δ, _probe_size, size(w, 1))) : @thunk(NNlib.∇conv_filter(x, Δ, cdims, kw...))
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
