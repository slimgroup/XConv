using Flux, XConv, BSON, IterTools
using Flux: onecold, logitcrossentropy, @epochs
using CUDA
using Parameters: @with_kw
using Distributed

addprocs(length(devices()))
@everywhere using CUDA, BSON, XConv, Flux
@everywhere using Flux: onecold, logitcrossentropy, @epochs
@everywhere using Parameters: @with_kw

@everywhere CUDA.unsafe_free!(h::Flux.OneHotMatrix) = CUDA.unsafe_free!(h.data)

cd(dirname(@__FILE__))

wp = default_worker_pool()

if CUDA.has_cuda()
    @everywhere device = gpu
    @everywhere CUDA.allowscalar(false)
    @everywhere iterator = CuIterator
    @info "Training on GPU"
else
    @everywhere iterator = ()
    @everywhere device = cpu
    @info "Training on CPU"
end

@everywhere include("./datautils.jl")
@everywhere include("./modelutils.jl")

@everywhere @with_kw mutable struct Args
    name::String = "MNIST"
    batchsize::Int = 128
    η::Float64 = 3e-3
    epochs::Int = 20
    splitr_::Float64 = 0.1
    probe_size::Int = 8
    mode::String = "TrueGrad"
    savepath::String = "./$(name)_bench"
end

@everywhere function accuracy(test_data, m)
    correct, total = 0, 0
    for (x, y) in iterator(test_data)
        correct += sum(onecold(cpu(m(x)), 0:9) .== onecold(cpu(y), 0:9))
        total += size(y, 2)
    end
    test_accuracy = correct / total
    test_accuracy
end

@everywhere augment(x) = x .+ device(0.1f0*randn(eltype(x), size(x)))
@everywhere filename(args) = joinpath(args.savepath, "$(args.name)_conv_$(args.mode)_$(args.batchsize)_$(args.probe_size).bson")

@everywhere function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
    isfile(filename(args)) && return
    # Load the train, validation data 
    train_data, val_data = get_train_data(args)
    test_data = get_test_data(args)

    @info("Constructing Model")	
    # Defining the loss and accuracy functions
    m = build_model(args.name) |> device
    loss(x, y) = logitcrossentropy(m(x), y)

    @info("Training $(args.name) with batch size $(args.batchsize)")
    args.mode == "TrueGrad" ? @info("Mode: $(args.mode)") : @info("Mode: $(args.mode), probing vectors: $(args.probe_size)")
    lhist = []
    
    # Defining the optimizer
    opt = ADAM(args.η)
    ps = Flux.params(m)

    # Initialize gradient mode
    XConv.initXConv(args.probe_size, args.mode)
    # Starting to train models
    for epoch in 1:args.epochs
        Base.flush(Base.stdout)
        local l
        for (x, y) in iterator(train_data)
            gs = Flux.gradient(ps) do 
                l = loss(x, y)
                l
            end
            Flux.update!(opt, ps, gs)
            push!(lhist, l)
        end

        validation_loss = 0f0
        for (x, y) in iterator(val_data)
            validation_loss += loss(x, y)
        end
        validation_loss /= length(val_data)
        acc = accuracy(val_data, m)
        @info "Epoch $epoch validation with ($(args.mode), $(args.probe_size)) loss = $(validation_loss), accuracy = $(acc)"
    end
    acc = accuracy(test_data, m)
    BSON.@save filename(args) params=cpu.(params(m)) acc lhist
    GC.gc()
end



datasets = ["MNIST", "CIFAR10"]

b_sizes = [2^i for i=5:10]
ps_sizes = [0..., [2^i for i=1:6]...]

# assign devices
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end

pw = default_worker_pool()

@sync for d in datasets
    for b in b_sizes
        for ps in ps_sizes
	    η = d == "CIFAR10" ? 3e-4 : 3e-3
            @async remotecall_fetch(train, pw;η=η, epochs=20,batchsize=b, probe_size=ps, name=d, mode= ps>0 ? "EVGrad" : "TrueGrad")
        end
    end
end
