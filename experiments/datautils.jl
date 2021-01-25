using Flux
using Flux: onehotbatch
using MLDatasets
using MLDataPattern: splitobs

get_test_data(args) = @eval $(Symbol(args.name, :_testdata))(;b_size=$args.batchsize, splitr_=$args.splitr_)
get_train_data(args) = @eval $(Symbol(args.name, :_traindata))(;b_size=$args.batchsize, splitr_=$args.splitr_)

function CIFAR10_traindata(;b_size=32, splitr_=.1)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    x, y = CIFAR10.traindata(Float32)

    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-splitr_)

    train_y = onehotbatch(train_y, 0:9)
    val_y = onehotbatch(val_y, 0:9)

    train = Flux.Data.DataLoader((train_x, train_y), batchsize=b_size, shuffle=true)
    val = Flux.Data.DataLoader((val_x, val_y), batchsize=b_size)
    
    return train, val
end

function CIFAR10_testdata(;b_size=32, splitr_=.1)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    test_x, test_y = CIFAR10.testdata(Float32)
    test_y = onehotbatch(test_y, 0:9)
    
    return Flux.Data.DataLoader((test_x, test_y), batchsize=b_size)
end


function MNIST_traindata(;b_size=32, splitr_=.1)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    x, y = MNIST.traindata(Float32)
    x = reshape(x, 28, 28, 1, :)
    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-splitr_)

    train_y = onehotbatch(train_y, 0:9)
    val_y = onehotbatch(val_y, 0:9)

    train = Flux.Data.DataLoader((train_x, train_y), batchsize=b_size, shuffle=true)
    val = Flux.Data.DataLoader((val_x, val_y), batchsize=b_size)
    
    return train, val
end

function MNIST_testdata(;b_size=32, splitr_=.1)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    test_x, test_y = MNIST.testdata(Float32)
    test_y = onehotbatch(test_y, 0:9)
    
    return Flux.Data.DataLoader((test_x, test_y), batchsize=b_size)
end