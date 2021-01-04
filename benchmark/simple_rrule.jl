using XConv, LinearAlgebra, PyPlot, Flux, BenchmarkTools, Statistics
BLAS.set_num_threads(4)

nx = 128
ny = 128
n_in = 2
n_out = 2

close("all")
fig, axs = subplots(3, 2, figsize=(10, 5))
title("Gradient comparisons \n N=$(nx)x$(ny), nchi=$(n_in), ncho=$(n_out)")

for i=1:6
    batchsize= 2^i
    stride = 1
    n_bench = 5
    nw   = 3;

    X = randn(Float32, nx, ny, n_in, batchsize);
    Y0 = randn(Float32, nx, ny, n_out, batchsize);

    # Flux network
    C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)
    w = C.weight

    XConv.initXConv(0, "TrueGrad")
    @btime g1 = gradient(w->.5*norm(conv($X, w;pad=1)- $Y0), $w);

    XConv.initXConv(2^3, "EVGrad")
    @btime g2 = gradient(w->.5*norm(conv($X, w;pad=1)- $Y0), $w);

    XConv.initXConv(0, "TrueGrad")
    @time g1 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]

    XConv.initXConv(2^3, "EVGrad")
    @time g2 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]
    nn = 100
    all = hcat([vec(gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]) for i=1:nn]...)
    gm = median(all, dims=2)[:, 1]
    gm2 = mean(all, dims=2)[:, 1]
    st = std(all; dims=2)[:, 1]

    t = 1:length(vec(g2))
    axsl = axs[i]
    axsl.set_title("batchsize=$(batchsize)")
    axsl.plot(t, vec(all[:, 1]), label="sample")
    axsl.plot(t, vec(gm), label="median")
    axsl.plot(t, vec(gm2), label="mean")
    axsl.fill_between(t, gm - st, gm + st, facecolor="orange", alpha=0.5, label="std($nn samples)")
    axsl.plot(t, vec(g1), label="true")
    i==6 && axsl.legend()
end