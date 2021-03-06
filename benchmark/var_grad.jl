using XConv, LinearAlgebra, PyPlot, Flux, BenchmarkTools, Statistics
using MLDatasets

BLAS.set_num_threads(Sys.CPU_THREADS)

function getbatch_CIFAR10(bsize, nchin, nchout; gaussian=false)
    X, _ = CIFAR10.traindata();
    gaussian && (X0 = randn(Float32, 32, 32, 3*nchin, bsize))
    !gaussian && (X0 = reshape(X[:, :, :, rand(1:50000, bsize*nchin)], 32, 32, 3*nchin, bsize))
    Y0 = reshape(X[:, :, :, rand(1:50000, bsize*nchout)], 32, 32, 3*nchout, bsize)
    return Float32.(X0), Float32.(Y0)
end

function getbatch_MNIST(bsize, nchin, nchout;gaussian=false)
    X, _ = MNIST.traindata();
    gaussian && (X0 = randn(Float32, 28, 28, nchin, bsize))
    !gaussian && (X0 = reshape(X[:, :, rand(1:60000, 3*bsize*nchin)], 28, 28, 3*nchin, bsize))
    Y0 = reshape(X[:, :, rand(1:60000, 3*bsize*nchout)], 28, 28, 3*nchout, bsize)
    return Float32.(X0), Float32.(Y0)
end

function qi(grads, level=.95)
    qis = [quantile(abs.(grads[i, :].- mean(grads[i, :])), level) for i=1:size(grads, 1)]
    return mean(grads; dims=2)[:, 1] .- vec(qis), mean(grads; dims=2)[:, 1] .+ vec(qis)
end

n_in = 1
n_out = 1
nx = 32
ny = 32

diffgrad = Array{Any}(undef, 4, 6)

close("all")
for ps=1:4
    fig, axs = subplots(3, 2, figsize=(10, 5), sharex=true, sharey=true)
    fig.suptitle("Gradient comparisons \n N=$(nx)x$(ny), nchi=$(3*n_in), ncho=$(3*n_out), prob_size=$(2^(ps+1))")

    for i=1:6
        batchsize= 2^(i-1)
        stride = 1
        n_bench = 5
        nw   = 3;

        # X, Y0 = getbatch_MNIST(batchsize, n_in, n_out)
        X, Y0 = getbatch_CIFAR10(batchsize, n_in, n_out; gaussian=true)
        # Flux network
        C = Conv((nw, nw), 3*n_in=>3*n_out, identity;pad=1, stride=stride)
        w = C.weight

        XConv.initXConv(0, "TrueGrad")
        @time g1 = gradient(w->.5f0*norm(conv(X, w;pad=1) - Y0), w)[1]

        XConv.initXConv(2^(ps+1), "EVGrad")
        @time g2 = gradient(w->.5f0*norm(conv(X, w;pad=1) - Y0), w)[1]
        nn = 100
        all_g = hcat([vec(gradient(w->.5f0*norm(conv(X, w;pad=1)- Y0), w)[1]) for i=1:nn]...)
        gm2 = mean(all_g, dims=2)[:, 1]
        st = std(all_g; dims=2)[:, 1]
        qtl, qtr = qi(all_g, .95)

        t = 1:length(vec(g2))
        axsl = axs[i]
        axsl.set_title("batchsize=$(batchsize)")
        axsl.plot(t, vec(all_g[:, 1]), label="sample")
        axsl.plot(t, vec(gm2), label="mean")
        axsl.fill_between(t, gm2 - st, gm2 + st, facecolor="orange", alpha=0.75, label="std($nn samples)")
        axsl.fill_between(t, qtl, qtr, facecolor="cyan", alpha=0.5, label="95% of values")
        axsl.plot(t, vec(g1), label="true")
        diffgrad[ps, i] = 1 .- vec(gm2)./vec(g1)
    end
    lines, labels = fig.axes[end].get_legend_handles_labels()
    fig.legend(lines, labels, loc = "upper left")
    tight_layout()
    # savefig("./benchmark/var_grad$(ps)_CIFAR10-randX.png", bbox_inches="tight")
end

fig, axs = subplots(3, 2, figsize=(10, 5), sharex=true, sharey=true)
title("Errors 1 - mean/true")
for i=1:6
    for ps=1:4
        axs[i].plot(diffgrad[ps, i], label="probe_size=$(2^(ps+1))")
    end
    axs[i].set_title("batch_size=$(2^(i-1))")
    axs[i].set_ylim([-1, 1])
end
lines, labels = fig.axes[end].get_legend_handles_labels()
fig.legend(lines, labels, loc = "upper left")
tight_layout()
# savefig("./benchmark/err_CIFAR10-randX.png", bbox_inches="tight")
