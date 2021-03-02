# Memroy efficient convolution layer via matrix sketching

This software provides the implementation of convolution layers where the gradient with respect to the weights
is approximated by an unbiased estimate. This estimate is obtained via matrix probing. This package contains two implementation:

- A `julia` implementation that overloads [NNlib](https://github.com/FluxML/NNlib.jl) for the computation of ``∇conv_filter``.
- A [PyTorch](https://pytorch.org/) implementation that defines a new convolution layer ``Xconv2D, Xconv3D``.


## Julia installation

To install the julia package, you can install it via the standard `dev` command

```julia
>> ]dev https://github.com/slimgroup/XConv
```

## Pip installation

The python source of this package can also be directly install via pip:


```bash
pip install git+https://github.com/slimgroup/XConv
```
or if you wish to get access to the experiments and benchmarking script:

```bash
git clone https://github.com/slimgroup/XConv
cd XConv
pip install -e .
```

This installation will install the default `torch`, we recommend to install the version that is best suited for your system following [Torch Installation](https://pytorch.org/get-started/locally/).


# Authors

This package is developpend at Georgia Institute of Technology byt the ML4Seismic Lab. The main autors of this package are:

- Mathias Louboutin: mlouboutin3@gatech.edu
- Ali Siahkoohi

# License

This package is distributed under the MIT license. Please check the LICENSE file for usage.
