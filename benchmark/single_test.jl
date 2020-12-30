using BenchmarkTools, XConv, LinearAlgebra, Flux, PyPlot, CUDA
CUDA.allowscalar(false)

nx = 256
ny = 256
b=10
n_in = 2
n_out = 2
stride = 1
n_bench = 5
nw   = 3;

# Flux network
C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)
X = rand([-1f0, 1f0], nx, ny, n_in, b)
Y = C(X)

p = Flux.params(C)

g1 = gradient(()->.5f0*norm(C(X)), p).grads[p[1]]
g21 = grad_ev(X, Y, 10, nw, stride);

@btime g1 = gradient(()->.5f0*norm(C(X)), p).grads[p[1]]
@btime g21 = grad_ev(X, Y, 10, nw, stride);


###### CUDA

C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride) |> gpu
X = rand([-1f0, 1f0], nx, ny, n_in, b) |> gpu
Y = C(X)

p = Flux.params(C)

g1 = gradient(()->.5f0*norm(C(X)), p).grads[p[1]]
g21 = grad_ev(X, Y, 10, nw, stride);

@btime g1 = gradient(()->.5f0*norm(C(X)), p).grads[p[1]]
@btime g21 = grad_ev(X, Y, 10, nw, stride);
