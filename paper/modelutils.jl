using Flux

function build_model(name="MNIST"; kw...)
    return @eval $(Symbol(:build_, name))(;$kw...)
end

# Build model
function build_MNIST(; imgsize = (28,28, 1), nclasses = 10)
    cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))	

    return Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(prod(cnn_output_size), 10))
end


# VGG16
function build_CIFAR10()
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