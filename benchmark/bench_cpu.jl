using NNlib, XConv, SharedArrays, LinearAlgebra, BenchmarkTools, CUDA
initXConv(0, "TrueGrad")
BenchmarkTools.DEFAULT_PARAMETERS.samples=10

# CUDA
CUDA.allowscalar(false)

#
sizes = [2^i for i=5:10]
batchsizes = [2^i for i=2:6]
channels = [4, 8]
ps = [4, 8, 16, 32]
nw   = 3;

t_nnlib = zeros(length(sizes), length(batchsizes), length(channels))
t_ev = zeros(length(sizes), length(batchsizes), length(channels), length(ps))

for (i,s)=enumerate(sizes)
    for (j, b)=collect(enumerate(batchsizes))
        for (k, c)=enumerate(channels)
            # X/Y
            X = CUDA.randn(Float32, s, s, c, b)
            Y0 = CUDA.randn(Float32, s, s, c, b)
            w = CUDA.randn(Float32, 3, 3, c, c)
            # conv
            cdims = cu(NNlib.DenseConvDims(X, w;padding=1))
            Δ = conv(X, w, cdims) - Y0
            g1 = ∇conv_filter(X, Δ, cdims)
            # NNlib
            t_nnlib[i,j,k] = 1f-6*mean(@benchmark g1 = ∇conv_filter($X, $Δ, $cdims)).time
            for (l, p)=enumerate(ps)
                g2 = grad_ev(X, Δ, p, nw, 1)
                tt = @elapsed begin
                    # with memory
                    t_ev[i,j,k,l] = 1f-6*mean(@benchmark g2 = grad_ev($X, $Δ, $p, $nw, 1)).time
                end
                @info "GPU: $(tt)sec for $p probing vector, $c channels, batchise of $b and image size of $s x $s"
                @info "NNlib: $(t_nnlib[i,j,k])ms, EV: $(t_ev[i,j,k,l])ms"
            end
        end
    end
end

