# Classifies MNIST digits with a convolutional network.
# Writes out saved model to the file "mnist_conv.bson".
# Demonstrates basic model construction, training, saving,
# conditional early-exit, and learning rate scheduling.
#
# This model, while simple, should hit around 99% test
# accuracy after training for approximately 20 epochs.

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy
using XConv
using Base.Iterators: partition
using Printf, BSON
using Parameters: @with_kw
using CUDA
using MLDatasets: CIFAR10
using MLDataPattern: splitobs

if has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end


@with_kw mutable struct Args
    batch_size::Int = 128
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 20
    splitr_::Float64 = 0.1
    savepath::String = "./cifar10_bsize/"
end


function get_processed_data(args; b=nothing)
    x, y = CIFAR10.traindata()

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.splitr_)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)
    
    return (train_x, train_y), (val_x, val_y)
end

function get_test_data()
    test_x, test_y = CIFAR10.testdata()
   
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    
    return test_x, test_y
end

# VGG16 and VGG19 models
function vgg16()
    Chain(
        Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(64),
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(128),
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(256),
        MaxPool((2,2)),
        Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        MaxPool((2,2)),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
        BatchNorm(512),
        MaxPool((2,2)),
        flatten,
        Dense(512, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, 10)
    )
end

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

function accuracy(test_data, m)
    correct, total = 0, 0
    for (x, y) in test_data
        correct += sum(onecold(cpu(m(x)), 0:9) .== onecold(cpu(y), 0:9))
        total += size(y, 2)
    end
    test_accuracy = correct / total
    test_accuracy
end

function train(mode; ps=0, kws...)
    args = Args(; kws...)

    @info("Loading data set")
    train, val = get_processed_data(args; b=args.batch_size)
    train_set = Flux.Data.DataLoader(train, batchsize=args.batch_size)
    test_set = Flux.Data.DataLoader(val, batchsize=args.batch_size)

    # Define our model.  We will use a simple convolutional architecture with
    # three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense layer.
    @info("Building model...")
    model = vgg16()
    # Load model and datasets onto GPU, if enabled
    train_set = gpu.(train_set)
    test_set = gpu.(test_set)
    model = gpu(model)
    @show accuracy(test_set, model)

    # `loss()` calculates the crossentropy loss between our prediction `y_hat`
    # (calculated from `model(x)`) and the ground truth `y`.  We augment the data
    # a bit, adding gaussian random noise to our image to make it more robust.
    loss(x, y) = logitcrossentropy(model(x), y)
    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    opt = ADAM(args.lr)
	
    @info("Beginning training loop... with batch_size $(args.batch_size) and $(ps) probing vectors")
    @show XConv._params
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:args.epochs
        XConv.initXConv(ps, mode)
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)
	    
        # Calculate accuracy:
        acc = accuracy(test_set, model)
		
        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end
	
        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            BSON.@save joinpath(args.savepath, "cifar10_conv_$(mode)_$(args.batch_size)_$(ps).bson") params=cpu.(params(model)) epoch_idx acc
            best_acc = acc
            last_improvement = epoch_idx
        end
	
        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10.0
            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end
	
        if epoch_idx - last_improvement >= 10
            #@warn(" -> We're calling this converged.")
            break
        end
    end
end

# Testing the model, from saved model
function test(mode; ps=0, kws...)
    args = Args(; kws...)
    
    # Loading the test data
    _, val = get_processed_data(args)
    test_set = Flux.Data.DataLoader(val, batchsize=args.batch_size)

    # Re-constructing the model with random initial weights
    model = vgg16()

    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "cifar10_conv_$(mode)_$(args.batch_size)_$(ps).bson") params
    
    # Loading parameters onto the model
    Flux.loadparams!(model, params)
    
    test_set = gpu.(test_set)
    model = gpu(model)
    @show accuracy(test_set, model)
end

cd(@__DIR__)

b_sizes = [2^i for i=5:10]
ps_sizes = [2^i for i=1:6]

for b in b_sizes
    train("TrueGrad";epochs=1, batch_size=b)
    @time train("TrueGrad"; batch_size=b)
    test("TrueGrad";batch_size=b)
    for ps in ps_sizes
        train("EVGrad"; epochs=1, batch_size=b, ps=ps)
        @time train("EVGrad"; batch_size=b, ps=ps)
        test("EVGrad"; batch_size=b, ps=ps)
    end
end
