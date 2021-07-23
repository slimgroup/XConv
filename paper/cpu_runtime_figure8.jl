using NNlib, XConv, LinearAlgebra, BenchmarkTools, PyPlot
initXConv(0, "TrueGrad")
BenchmarkTools.DEFAULT_PARAMETERS.samples=10

# BLAS param
nthreads = Sys.CPU_THREADS
BLAS.set_num_threads(nthreads)

if !isfile("bench_cpu.jld")

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
                    g2 = grad_ev(X, Δ, p, nw)
                    tt = @elapsed begin
                        # with memory
                        t_ev[i,j,k,l] = 1f-6*mean(@benchmark g2 = grad_ev($X, $Δ, $p, $nw)).time
                    end
                    @info "CPU: $(tt)sec for $p probing vector, $c channels, batchise of $b and image size of $s x $s"
                    @info "NNlib: $(t_nnlib[i,j,k])ms, EV: $(t_ev[i,j,k,l])ms"
                end
            end
        end
    end
    @save "bench_cpu.jld" t_ev t_nnlib hannels batchsizes sizes ps

else
    @load "bench_cpu.jld" t_ev t_nnlib channels batchsizes sizes ps
end


colrs = ["--*r", "--*k", "--*c", "--*g"]
count = 1
for (i,c)=enumerate(channels)
    for (k, b)=enumerate(batchsizes)
        # Remove ot of memory point
        (c==32 && b == 64) ? li = length(sizes)-1 : li=length(sizes)
        figure(count, figsize=(10, 5))
        global count +=1
        xlabel("N_x", fontsize=18)
        ylabel("Runtime (ms)", fontsize=18)
        xticks(fontsize=18)
        yticks(fontsize=18)
        for (l,p)=enumerate(ps)
            loglog(sizes[1:li], t_ev[1:li, k, i, l], label="r=$p", colrs[l], markersize=3, linewidth=1,basex=2, basey=10)
        end
        loglog(sizes[1:li], t_nnlib[1:li, k, i], "--ob", label="True", markersize=3, linewidth=1,basex=2, basey=10)
        legend(loc="upper left", fontize=18)
        tight_layout()
        savefig("./bench_cpu_$(c)_$(b).png", bbox_inches="tight")
    end
end
