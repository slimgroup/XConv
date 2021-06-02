using Flux, XConv, BSON, IterTools
using Flux: onecold, logitcrossentropy, @epochs
using CUDA
using Parameters: @with_kw
using ProgressMeter

CUDA.unsafe_free!(h::Flux.OneHotMatrix) = CUDA.unsafe_free!(h.data)

cd(dirname(@__FILE__))

if CUDA.has_cuda()
    device = gpu
    CUDA.allowscalar(false)
    iterator = CuIterator
    @info "Training on GPU"
else
    iterator = ()
    device = cpu
    @info "Training on CPU"
end

include("./datautils.jl")
include("./modelutils.jl")

@with_kw mutable struct Args
    name::String = "MNIST"
    batchsize::Int = 128
    η::Float64 = 3e-3
    η_fact::Float64 = .9
    progress_fact::Float64 = .001
    epochs::Int = 20
    splitr_::Float64 = 0.1
    probe_size::Int = 8
    mode::String = "TrueGrad"
    savepath::String = "./$(name)_bench"
end

function accuracy(test_data, m)
    correct, total = 0, 0
    for (x, y) in iterator(test_data)
        correct += sum(onecold(cpu(m(x)), 0:9) .== onecold(cpu(y), 0:9))
        total += size(y, 2)
    end
    test_accuracy = correct / total
    test_accuracy
end

augment(x) = x .+ device(0.1f0*randn(eltype(x), size(x)))
filename(args) = joinpath(args.savepath, "$(args.name)_conv_$(args.mode)_$(args.batchsize)_$(args.probe_size).bson")

function check(fname)
    isfile(fname) ? (BSON.@load fname acc) : (return false)
    acc < .1 ? rm(fname) : (return true)
    false
end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
    check(filename(args)) && return

    # Load the train, validation data 
    train_data, val_data = get_train_data(args)
    test_data = get_test_data(args)

    @info("Constructing Model")	
    # Defining the loss and accuracy functions
    m = build_model(args.name) |> device
    loss(x, y) = logitcrossentropy(m(x), y)

    @info("Training $(args.name) with batch size $(args.batchsize)")
    lhist = []
    
    # Defining the optimizer
    opt = ADAM(args.η)
    ps = Flux.params(m)

    # Initialize gradient mode
    XConv.initXConv(args.probe_size, args.mode)
    # Starting to train models
    p = Progress(length(train_data) * args.epochs)
    best = 0
    for epoch in 1:args.epochs
        Base.flush(Base.stdout)
        local l, acc
	acc = accuracy(val_data, m)
	n = length(train_data)
        for (x, y) in iterator(train_data)
            gs = Flux.gradient(ps) do 
                l = loss(x, y)
                l
            end
            Flux.update!(opt, ps, gs)
            push!(lhist, l)
	    ProgressMeter.next!(p; showvalues = [(:loss, l), (:epoch, epoch), (:Mode, args.mode), (:ps, args.probe_size), (:η, args.η), (:accuracy, acc)])
	end

        validation_loss = 0f0
        for (x, y) in iterator(val_data)
            validation_loss += loss(x, y)
        end
        validation_loss /= length(val_data)
        acc_loc = accuracy(val_data, m)
	(acc_loc/acc - 1) < args.progress_fact && (args.η *= args.η_fact)
        @info "Epoch $epoch validation with $(XConv._params) loss = $(validation_loss), accuracy = $(acc_loc)"
    end
    acc = accuracy(test_data, m)
    BSON.@save filename(args) params=cpu.(params(m)) acc lhist
    GC.gc()
end

b_sizes = [2^i for i=5:11]
ps_sizes = [0..., [2^i for i=1:7]...]


for b in b_sizes
    for ps in ps_sizes
        η = 3e-3
        train(;η=η, epochs=10+2*log2(b),batchsize=b, probe_size=ps, name=d, mode= ps>0 ? "EVGrad" : "TrueGrad")
    end
end


nps, nb = length(ps_sizes)+1, length(b_sizes)
accs = Dict("MNIST" => zeros(nps, nb))
d = "MNIST"

for (i,b) in enumerate(b_sizes)
    @show b, d
    BSON.@load joinpath("./$(d)_bench/", "$(d)_conv_TrueGrad_$(b)_0.bson") acc
    accs[d][1, i] = acc
    for (j,ps) in enumerate(ps_sizes)
        BSON.@load joinpath("./$(d)_bench/", "$(d)_conv_EVGrad_$(b)_$(ps).bson") acc
        accs[d][j+1, i] = acc
    end
end


# ppri
header(i) = i==1 ? "| default   |" : "| ``ps=$(ps_sizes[i-1])`` |"

str_mnist = prod([header(j)*prod([" ``$(round(aa, digits=4))`` |" for aa in accs["MNIST"][j,:]])*"\n" for j=1:nps])

top = "|           |"*prod(["``B=$b``   |" for b in b_sizes])*"\n|"*"|:---------:"^(nb+1)*"|\n"

println("MNIST")
println(top*str_mnist)
