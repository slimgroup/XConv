using XConv, LinearAlgebra, PyPlot, Flux, BenchmarkTools, Statistics
BLAS.set_num_threads(4)

function qi(grads, level=.95)
    qis = [quantile(abs.(grads[i, :].- mean(grads[i, :])), level) for i=1:size(grads, 1)]
    return mean(grads; dims=2)[:, 1] .- vec(qis), mean(grads; dims=2)[:, 1] .+ vec(qis)
end

nx = 128
ny = 128
n_in = 2
n_out = 2

close("all")
for ps = [2^i for i=2:5]
    fig, axs = subplots(3, 2, figsize=(10, 5), sharex=true, sharey=true)
    fig.suptitle("Gradient comparisons \n N=$(nx)x$(ny), nchi=$(n_in), ncho=$(n_out), prob_size=$(ps)")

    for i=1:6
        batchsize= 2^(i-1)
        stride = 1
        n_bench = 5
        nw   = 3;

        X = randn(Float32, nx, ny, n_in, batchsize);
        Y0 = randn(Float32, nx, ny, n_out, batchsize);

        # Flux network
        C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)
        w = C.weight

        # XConv.initXConv(0, "TrueGrad")
        # @btime g1 = gradient(w->.5*norm(conv($X, w;pad=1)- $Y0), $w);

        # XConv.initXConv(ps, "EVGrad")
        # @btime g2 = gradient(w->.5*norm(conv($X, w;pad=1)- $Y0), $w);

        XConv.initXConv(0, "TrueGrad")
        @time g1 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]

        XConv.initXConv(ps, "EVGrad")
        @time g2 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]
        nn = 100
        all_g = hcat([vec(gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]) for i=1:nn]...)
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
    end
    lines, labels = fig.axes[end].get_legend_handles_labels()
    fig.legend(lines, labels, loc = "upper left")
    tight_layout()
    savefig("./benchmark/var_grad$(ps).png", bbox_inches="tight")
end