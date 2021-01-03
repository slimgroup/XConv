using XConv, LinearAlgebra, PyPlot, Flux, BenchmarkTools
BLAS.set_num_threads(4)

nx = 128
ny = 128
batchsize= 32
n_in = 4
n_out = 4
stride = 1
n_bench = 5
nw   = 3;

X = randn(Float32, nx, ny, n_in, batchsize);
Y0 = randn(Float32, nx, ny, n_out, batchsize);

# Flux network
C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)
w = C.weight

XConv.initXConv(0, "TrueGrad")
@btime g1 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w);

XConv.initXConv(2^3, "EVGrad")
@btime g2 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w);

XConv.initXConv(0, "TrueGrad")
@time g1 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]

XConv.initXConv(2^3, "EVGrad")
@time g2 = gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]
nn = 100
all = hcat([vec(gradient(w->.5*norm(conv(X, w;pad=1)- Y0), w)[1]) for i=1:nn]...)
gm = median(all, dims=2)[:, 1]
gm2 = mean(all, dims=2)[:, 1]
gmin = minimum(all, dims=2)[:, 1]
gmax = maximum(all, dims=2)[:, 1]

t = 1:length(vec(g2))
figure()
plot(t, vec(all[:, 10]), label="sample")
plot(t, vec(gm), label="median")
plot(t, vec(gm2), label="mean")
fill_between(t, gmin, gmax, facecolor="cyan", alpha=0.75, label="min-max")
plot(t, vec(g1), label="true")
legend()
