using NNlib, XConv, LinearAlgebra, BenchmarkTools, CUDA
initXConv(0, "TrueGrad")
BenchmarkTools.DEFAULT_PARAMETERS.samples=10

# CUDA
CUDA.allowscalar(false)
# BLAS param
nthreads = div(Sys.CPU_THREADS, 2)
BLAS.set_num_threads(nthreads)

#
cuda = ARGS["CUDA"]

sizes = [2^i for i=5:10]
batchsizes = [2^i for i=2:6]
channels = [4, 32]
ps = [4, 8, 16, 32]
nw   = 3;

t_nnlib = zeros(length(sizes), length(batchsizes), length(channels))
t_ev = zeros(length(sizes), length(batchsizes), length(channels), length(ps))

for (i,s)=enumerate(sizes)
    for (j, b)=enumerate(batchsizes)
        for (k, c)=enumerate(channels)
            # X/Y
            X = randn(Float32, s, s, c, b)
            Y0 = randn(Float32, s, s, c, b)
            w = randn(Float32, 3, 3, c, c)
            # conv
            cdims = NNlib.DenseConvDims(X, w;padding=1)
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
                @info "CPU: $(tt)sec for $p probing vector, $c channels, batchise of $b and image size of $s x $s"
                @info "NNlib: $(t_nnlib[i,j,k])ms, EV: $(t_ev[i,j,k,l])ms"
            end
        end
    end
end

colrs = ["--*r", "--*k", "--*c", "--*g"]
count = 1
for (i,c)=enumerate(channels)
    for (k, b)=enumerate(batchsizes)
        # Remove ot of memory point
        (c==32 && b == 64) ? li = length(sizes)-1 : li=length(sizes)
        figure(count, figsize=(10, 5))
        global count +=1
        xlabel("N_x")
        ylabel("Runtime (ms)")
        for (l,p)=enumerate(ps)
            loglog(sizes[1:li], t_ev[1:li, k, i, l], label="probe-$p", colrs[l], markersize=3, linewidth=1,basex=2, basey=10)
        end
        loglog(sizes[1:li], t_nnlib[1:li, k, i], "--ob", label="NNlib", markersize=3, linewidth=1,basex=2, basey=10)
        legend(loc="upper left")
        tight_layout()
        savefig("./paper/figures/runtimes/bench_gpu_$(c)_$(b).png", bbox_inches="tight")
    end
end
