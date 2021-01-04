using XConv, LinearAlgebra, Flux, PyPlot, BenchmarkTools

BLAS.set_num_threads(2)

nx = 256
ny = 256
batchsize=10
n_in = 1
n_out = 1
stride = 1
n_bench = 10
nw   = 3;

# Flux network
C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)

batches = [2^k for k=0:8]

tf = zeros(length(batches))
t1 = similar(tf)
t10 = similar(tf)
t50 = similar(tf)
t100 = similar(tf)
angles = zeros(length(batches), 4)

close("all")

# Init plot
fig, axsl = subplots(3, 3, figsize=(10, 5))
title("Conv layer gradient chi-$(n_in), cho-$(n_out)")

for (i, b)=enumerate(batches)
    println("Gradient for batchsize=$b")

    local X = randn(Float32, nx, ny, n_in, b)
    local Y = C(X) - randn(Float32, nx, ny, n_out, b)

    cdims = DenseConvDims(X, C.weight; stride=C.stride, padding=C.pad, dilation=C.dilation)

    # Run once to get gradient
    @time local g1 = ∇conv_filter(X, Y, cdims)
    @time local g20 = grad_ev(X, Y, 5, nw, stride);
    @time local g21 = grad_ev(X, Y, 10, nw, stride);
    @time local g22 = grad_ev(X, Y, 50, nw, stride);
    @time local g23 = grad_ev(X, Y, 100, nw, stride);

    # Compute similarities
    angles[i, 1] = dot(g20, g1)/(norm(g20)*norm(g1))
    angles[i, 2] = dot(g21, g1)/(norm(g21)*norm(g1))
    angles[i, 3] = dot(g22, g1)/(norm(g22)*norm(g1))
    angles[i, 4] = dot(g23, g1)/(norm(g23)*norm(g1))

    # Benchmark runtime
    tf[i] = mean((@benchmark ∇conv_filter($X, $Y, $cdims) samples=n_bench).times)
    t1[i] = mean((@benchmark grad_ev($X, $Y, 5, $nw, $stride) samples=n_bench).times)
    t10[i] = mean((@benchmark grad_ev($X, $Y, 10, $nw, $stride) samples=n_bench).times)
    t50[i] = mean((@benchmark grad_ev($X, $Y, 50, $nw, $stride) samples=n_bench).times)
    t100[i] = mean((@benchmark grad_ev($X, $Y, 100, $nw, $stride) samples=n_bench).times)

    # Plot result
    axsl[i].plot(vec(g20)/norm(g20, Inf), label="LR(s=5)", "-r")
    axsl[i].plot(vec(g21)/norm(g21, Inf), label="LR(s=10)", "-b")
    axsl[i].plot(vec(g22)/norm(g22, Inf), label="LR(s=50)", "-g")
    axsl[i].plot(vec(g23)/norm(g23, Inf), label="LR(s=100))", "-c")
    axsl[i].plot(vec(g1)/norm(g1, Inf), label="Flux", "-k")
    axsl[i].set_title("batchsize=$b")
end
lines, labels = fig.axes[end].get_legend_handles_labels()
fig.legend(lines, labels, loc = "upper left")
tight_layout()

figure()
title("10 Gradient computation speedup")
plot(batches, tf, label="Flux")
plot(batches,t1, label="LR(s=5)")
plot(batches,t10, label="LR(s=10)")
plot(batches,t50, label="LR(s=50)")
plot(batches,t100, label="LR(s=100)")
xlabel("Batch size")
ylabel("Runtime(s)")
legend()

figure()
title("Similarty x' x2 / (||x|| ||x2||)")
plot(batches,angles[:, 1], label="LR(s=5)")
plot(batches,angles[:, 2], label="LR(s=10)")
plot(batches,angles[:, 3], label="LR(s=50)")
plot(batches,angles[:, 4], label="LR(s=100)")
legend()

@show angles
