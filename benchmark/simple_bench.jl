using XConv, LinearAlgebra, Flux, PyPlot

BLAS.set_num_threads(2)

nx = 256
ny = 256
batchsize=10
n_in = 2
n_out = 2
stride = 1
n_bench = 5
nw   = 3;

X = randn(Float32, nx, ny, n_in, batchsize);
Y = randn(Float32, nx, ny, n_out, batchsize);
g20 = grad_ev(X, Y, 5, nw, stride);
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
ni = min(n_in, 3)
no = min(n_out, 3)
figs = Array{Any}(undef, ni, no)
axs = Array{Any}(undef, ni, no)

for ci=1:ni
    for co=1:no
        figs[ci, co], axs[ci, co] = subplots(3, 3, figsize=(10, 5))
        figs[ci, co].suptitle("Conv layer gradient chi-$(ci), cho-$(co)")
    end
end

for (i, b)=enumerate(batches)
    println("Gradient for batchsize=$b")

    local X = rand([-1f0, 1f0], nx, ny, n_in, b)
    rn() = 0 #randn(Float32, nx, ny, n_in, b)*0.01f0*norm(vec(X))

    p = params(C)

    @time local g1 = gradient(()->.5*norm(C(X).+rn()), p).grads[p[1]]
    tf[i] = @elapsed begin
        for nn=1:n_bench
            local g1 = gradient(()->.5*norm(C(X).+rn()), p).grads[p[1]]
        end
    end

    tbase = @elapsed Y = C(X) .+ rn();
    tf[i] -= n_bench*tbase

    @time local g20 = grad_ev(X, Y, 5, nw, stride);
    @time local g21 = grad_ev(X, Y, 10, nw, stride);
    @time local g22 = grad_ev(X, Y, 50, nw, stride);
    @time local g23 = grad_ev(X, Y, 100, nw, stride);

    angles[i, 1] = dot(g20, g1)/(norm(g20)*norm(g1))
    angles[i, 2] = dot(g21, g1)/(norm(g21)*norm(g1))
    angles[i, 3] = dot(g22, g1)/(norm(g22)*norm(g1))
    angles[i, 4] = dot(g23, g1)/(norm(g23)*norm(g1))

    t1[i] = @elapsed begin
        for nn=1:n_bench
            local g21 = grad_ev(X, Y, 5, nw, stride);
        end
    end

    t10[i] = @elapsed begin
        for nn=1:n_bench
            local g21 = grad_ev(X, Y, 10, nw, stride);
        end
    end

    t50[i] = @elapsed begin
        for nn=1:n_bench
            local g22 = grad_ev(X, Y, 50, nw, stride);
        end
    end

    t100[i] = @elapsed begin
        for nn=1:n_bench
            local g23 = grad_ev(X, Y, 100, nw, stride);
        end
    end

    # Plot result
    for ci=1:ni
        for co=1:no
            local axsl = axs[ci, co][i]
            axsl.plot(vec(g20[:, :, ci, co])/norm(g20[:, :, ci, co], Inf), label="LR(s=5)", "-r")
            axsl.plot(vec(g21[:, :, ci, co])/norm(g21[:, :, ci, co], Inf), label="LR(s=10)", "-b")
            axsl.plot(vec(g22[:, :, ci, co])/norm(g22[:, :, ci, co], Inf), label="LR(s=50)", "-g")
            axsl.plot(vec(g23[:, :, ci, co])/norm(g23[:, :, ci, co], Inf), label="LR(s=100))", "-c")
            axsl.plot(vec(g1[:, :, ci, co])/norm(g1[:, :, ci, co], Inf), label="Flux", "-k")
            axsl.set_title("batchsize=$b")
            i==1 && axsl.legend()
        end
    end
end
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