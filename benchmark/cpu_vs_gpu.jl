using XConv, BenchmarkTools, LinearAlgebra, Flux, CUDA

# CUDA param
CUDA.allowscalar(false)
# BLAS param
nthreads = div(Sys.CPU_THREADS, 2)
BLAS.set_num_threads(nthreads)
XConv.initXConv(0, "TrueGrad")

nx = 256
ny = 256
b = 16
n_in = 16
n_out = 4
stride = 1
n_bench = 5
nw   = 3;

@info "Running runntime comparison between CPU and CPU for N=$(nx)x$(ny), batchsize=$b, $(n_in) input channels and $(n_out) output channels"

# Flux network
C = Conv((nw, nw), n_in=>n_out, identity;pad=1, stride=stride)
X = randn(Float32, nx, ny, n_in, b)
Y = C(X)
cdims = DenseConvDims(X, C.weight; padding=1)

@btime gcpu = NNlib.∇conv_filter($X, $Y, $cdims)
@btime gcpu_ev = grad_ev($X, $Y, 8, $nw, $stride);

@info "CPU benchmark with $(nthreads) threads"

for i=1:4
  @info "test $i"
  @time gcpu = NNlib.∇conv_filter(X, Y, cdims)
  @time gcpu_ev = grad_ev(X, Y, 8, nw, stride);

  XConv.initXConv(0, "TrueGrad")
  @time gradient(C->.5f0*norm(C(X)), C)
  XConv.initXConv(2^3, "EVGrad")
  @time gradient(C->.5f0*norm(C(X)), C)
  XConv.initXConv(0, "TrueGrad")
end

XConv.initXConv(0, "TrueGrad")
@btime gradient(C->.5f0*norm(C(X)), C)
XConv.initXConv(2^3, "EVGrad")
@btime gradient(C->.5f0*norm(C(X)), C)
XConv.initXConv(0, "TrueGrad")


###### CUDA

for dtype=[Float32]
    @info "GPU benchmark in $dtype"

    C = Conv(CUDA.randn(dtype, nw, nw, n_in, n_out), CUDA.zeros(dtype, n_out), identity;pad=1)
    X = CUDA.randn(dtype, nx, ny, n_in, b)
    Y = C(X)

    cdims = DenseConvDims(X, w; padding=1)

    XConv.initXConv(0, "TrueGrad")
    @btime ggpu = NNlib.∇conv_filter($X, $Y, $cdims)
    ggpu_ev = grad_ev(X, Y, 8, nw, stride)
    @btime ggpu_ev = grad_ev($X, $Y, 8, $nw, $stride);
    @btime ggpu_ev = grad_ev($X, $Y, 8, $nw, $stride);

    for i=1:4
        @info "test $i"
        XConv.initXConv(0, "TrueGrad")
        @time ggpu = NNlib.∇conv_filter(X, Y, cdims)
        @time ggpu_ev = grad_ev(X, Y, 8, nw, stride);

        XConv.initXConv(0, "TrueGrad")
        @time gradient(C->.5f0*norm(C(X)), C)
        XConv.initXConv(2^3, "EVGrad")
        @time gradient(C->.5f0*norm(C(X)), C)
        XConv.initXConv(0, "TrueGrad")
    end

    XConv.initXConv(0, "TrueGrad")
    @time gradient(C->.5f0*norm(C(X))^2, C)
    XConv.initXConv(2^3, "EVGrad")
    @time gradient(C->.5f0*norm(C(X))^2, C)
    XConv.initXConv(0, "TrueGrad")
    @btime gradient(C->.5f0*norm(C(X))^2, C)
    XConv.initXConv(2^3, "EVGrad")
    @btime gradient(C->.5f0*norm(C(X))^2, C)

end
