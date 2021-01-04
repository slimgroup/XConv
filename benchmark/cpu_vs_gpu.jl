using BenchmarkTools, XConv, CUDA, LinearAlgebra, NNlib, Flux
# CUDA param
CUDA.allowscalar(false)
# BLAS param
nthreads = div(Sys.CPU_THREADS, 2)
BLAS.set_num_threads(nthreads)

nx = 256
ny = 256
b = 32
n_in = 4
n_out = 4
stride = 1
n_bench = 5
nw   = 3;

@info "Running runntime comparison between CPU and CPU for N=$(nx)x$(ny), batchsize=$b, $(n_in) input channels and $(n_out) output channels"

# Flux network
C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)
X = rand([-1f0, 1f0], nx, ny, n_in, b)
Y = C(X)
w = C.weight
cdims = DenseConvDims(X, C.weight; padding=1)

gcpu = NNlib.∇conv_filter(X, Y, cdims)
gcpu_ev = grad_ev(X, Y, 8, nw, stride);

@info "CPU benchmark with $(nthreads) threads"

@btime gcpu = NNlib.∇conv_filter(X, Y, cdims)
@btime gcpu_ev = grad_ev(X, Y, 8, nw, stride);

XConv.initXConv(0, "TrueGrad")
@btime gradient(C->.5f0*norm(C(X)), C)
XConv.initXConv(2^3, "EVGrad")
@btime gradient(C->.5f0*norm(C(X)), C)
XConv.initXConv(0, "TrueGrad")
###### CUDA

@info "GPU benchmark"

C = gpu(C)
X = gpu(X)
Y = gpu(Y)
w = gpu(C.weight)

cdims = DenseConvDims(X, w; padding=1)

ggpu = NNlib.∇conv_filter(X, Y, cdims)
ggpu_ev = grad_ev(X, Y, 8, nw, stride);

@btime ggpu = NNlib.∇conv_filter(X, Y, cdims)
@btime ggpu_ev = grad_ev(X, Y, 8, nw, stride);

XConv.initXConv(0, "TrueGrad")
@btime gradient(C->.5f0*norm(C(X)), C)
XConv.initXConv(2^3, "EVGrad")
@btime gradient(C->.5f0*norm(C(X)), C)

