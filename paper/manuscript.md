---
title: Efficient unbiased backpropagation via matrix probing for filter updates.
author: |
  Mathias Louboutin^1^, Felix J. Herrmann^1^, Ali Siahkoohi^1^ \
  ^1^School of Computational Science and Engineering, Georgia Institute of Technology\
bibliography:
    - probeml.bib
---

# TODO

- [] Add refs
- [] Run gpu benchmark
- [] Cleanup theory
- [] Redo some plots

## Abstract

bonjour

## Introduction

- Convolution layer gradients are expensive and main cost of CNNs.
- Unbiased approximation shown to be good (put refs)
- Lessons learned from PDE adjoint state

## Theory

The backpropagation through a convolution layer is the correlation between the layer input ``X`` and the backpropagated residual ``\Delta``.

```math {#grad_im2col}
    \delta W[i,j] = X_{i,j} \times \Delta
```

where ``X_{i,j}`` is ``X`` shifted by ``i, j`` in the image space (``X[x-i, Y-j, channel, batch]``). This correlation is conventionally implemented with the *im2col+gemm* algorithm.

Another way to look at this update is to extract the trace of the outer product of ``X`` with ``\Delta`` at the offsets corresponding to the kernel indices, i.e the diagonals ``[-N_x-1, -N_x, -N_x+1, -1, 0, 1, N_x-1, N_x, N_x+1]`` for a 3x3 convolution. While forming this outer-product would be unefficient both computationnaly and memory-wise, probing techniques [refs] provide estimates of the trace via averaging matrix-vector products:

```math {#grad_ev}
    \tilde{\delta W[i,j]} = \frac{12}{M} \sum_{i=1}^M z_i^T X \Delta^T z_i,
```

where ``\tilde{X}, \tilde{\Delta}`` are ``X, \Delta`` vectorized along the image and channel dimensions and ``z_i`` are random probing vectors drawn from ``\mathcal{U}(-.5, .5)`` in our case. This sum is normalized by ``12`` to compensate for the variance of the uniform distribution. In theory, Radamaecher or ``\mathcal{N}(0, 1)`` provide better estimates of the trace, however, these distributions are a lot more expesnsive to draw from for large vectors and would impact the performance.

# Simplified performance estimates

We consider here the flop estimates for two cases of a convolution layer with kernel size ``K x K``. The standard dot-product based correlation of input and output, and the probing trace estimate for a single probing vector (``M=1`` in Eq. #grad_ev). We consider a signle channel input ``x \in \mathcal{R}^{N_x \times N_y \times 1 \times B}`` and single channel residual ``\Delta \in \mathcal{R}^{N_x \times N_y \times 1 \times B}`` where ``N`` is the number of pixels in each image dimension and ``B``is the batch size.

The simplest way to compute the standard gradient is to sum over batches the shifted correlation of input and output. Each of the correlation is a dot product that requires ``N=N_x N_y`` multiplications and ``N-1`` additions. This leads to the estimate:

```math {#flops-std}
Flops_{std} = K B (N + N - 1) \approx 2KBN
```

That is linear in the image dimension, kernel size and batchsize. The estimate of the trace corresponding to ``\delta W[i, j]`` via probing on the other hand reuqires three steps:

- 1. Multiply ``\Delta`` by probing the vector ``z_1``.
- 2. Multiply the result by ``X``.
- 3. Multiply by ``z_1^{i,j}`` that is ``z_1`` shifted by ``i+N_x*j`` to extract the offdiagonal trace.

We note that steps 1 and 2 are common to all offsets and only need to be computed once. Each of these steps requires a single matrix-vector product of respective sizes ``B \text{ by } N , N``, ``B \text{ by } N , B`` and ``N , N`` that sums up to:

```math {#flops-eiv}
Flops_{ev} &= B (N + N + N-1) + N (B + B-1) + K (N + N - 1) \\
      &= 3NB +  2BN + 2KN \\
      &= (5B + 2K) N
```

That is linear in ``N`` as well but affine in ``B`` and ``K``. From these two estimate, we can compute the Flop ratio between the two methods:

```math {#ratio}
  r = \frac{2KBN}{ (5B + 2K) N} = \frac{2KN}{2K + 2B}
```

This computatonal ratio is independent of the image size since both flop estimates are linear with respect ot i. We plot this ratio for varying batch size ``B`` and kernel size ``K`` in Figure #ratio-flops. We show there that, admitely for a very simple performance model, that probing method can be cheaper up to a factor of almost 50 for large batch size. This computational adavantage comes from the fact that the batch size is absorbed in the outer product of ``X`` and ``\Delta`` rather than in a loop.

#### Figure: {#ratio-flops}
![ratio](figures/ratio.png)
:Ratio ``\frac{2KB}{5B+2K}`` for varying stencil and batch size. High value mean EIV is more efficient (up to 40 times better for small stencil, high batch size)


# Experiments

In order to validate our method and provide a more rigorous evalation of its efficientcy, we compare our method agains [NNlib.jl](https://github.com/FluxML/NNlib.jl) [refs]. NNlib is an advanced librairie for CNNs that implements state of the art *im2col+gemm* on CPUs and interfaces to cuDNN on GPUs via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) that implements highly efficient optimized kernels. We consider the three folowing benchmarks:

- Accuracy. We look at the accuracy of the obtained gradient against the true gradient for varying batch size, image size and number of probing vectors.
- Biasing. We verify the the gradient is unbiased using the CIFAR10 dataset computing expectation of our gradient approximation against the true gradient.
- Pure performance. In this case we consider the computational runtime for a single convolution layer gradient for varying image size, batch size and number of channel. This benchmark is performed on CPU and GPU

## Accuracy and bias

We compute the gradient with respect to the filter of ``\frac{1}{2}||C(X) - Y||^2`` where ``C`` is a convolution layer ([FLux.jl](https://github.com/FluxML/NNlib.jl)) without bias and no activation and ``Y`` is a batc hof images from the CIFAR10 dataset. We consider two cases for ``X``. In the first case, ``X`` is a batch drawn from the CIFAR10 dataset and in the second case, ``X`` is a random variable drawn from ``\mathcal{N}(0, 1)``.

#### Figure: {#bias-cifar10-rand}
![probing size=4](figures/bias/var_grad1_CIFAR10-randX.png){width=50%}
![probing size=8](figures/bias/var_grad2_CIFAR10-randX.png){width=50%}
![probing size=16](figures/bias/var_grad3_CIFAR10-randX.png){width=50%}
![probing size=32](figures/bias/var_grad4_CIFAR10-randX.png){width=50%}
: Gradients for ``X`` drawn from the normal distribution. We can see than in expectation, the probed gradient corresponds to the true gradient making it unbiased. 


#### Figure: {#bias-cifar10}
![probing size=4](figures/bias/var_grad1_CIFAR10.png){width=50%}
![probing size=8](figures/bias/var_grad2_CIFAR10.png){width=50%}\
![probing size=16](figures/bias/var_grad3_CIFAR10.png){width=50%}
![probing size=32](figures/bias/var_grad4_CIFAR10.png){width=50%}
: Gradients for ``X`` drawn from the CIFAR10 dataset. We can see than in expectation, the probed gradient corresponds to the true gradient making it unbiased. 

We show these gradients on Figures #bias-cifar10 and #bias-cifar10-rand\. These figures demonstrates three points. First, we can see that an increasing number of probing vector leads to a better estimate of the gradient and a reduced standard deviation making a single sample a more accurate estimates. Second, we show that our estimate is unbiased as both the mean and mediam matches the the true gradient. Finally, these figures show that our gradient estimate is accurate and converges to the true gradient as the probing size increases, and we also show in Figure #batch-effect that our estimates is more accurate for larger batch sizes and/or larger images.

## Performance

We show on Figure #cpu-bench and #gpu-bench the benchmarked runtime to compute a single gradient with NNlib and with our method for varying image sizes and batch sizes. The benchmark was done for a small (4 =>4) and large number of channel (32 =>32).

#### Figure: {#cpu-bench}
![B=4](figures/runtimes/bench_cpu_4_4.png){width=40%}
![B=4](figures/runtimes/bench_cpu_32_4.png){width=40%}\
![B=8](figures/runtimes/bench_cpu_4_8.png){width=40%}
![B=8](figures/runtimes/bench_cpu_32_8.png){width=40%}\
![B=16](figures/runtimes/bench_cpu_4_16.png){width=40%}
![B=16](figures/runtimes/bench_cpu_32_16.png){width=40%}\
![B=32](figures/runtimes/bench_cpu_4_32.png){width=40%}
![B=32](figures/runtimes/bench_cpu_32_32.png){width=40%}\
![B=64](figures/runtimes/bench_cpu_4_64.png){width=40%}
![B=64](figures/runtimes/bench_cpu_32_64.png){width=40%}\
:CPU benchmark on a *Intel(R) Xeon(R) CPU E3-1270 v6 @ 3.80GHz* node. The left column contains the runtimes for 4 channels and the right column for 32 channels. We can see that for large images and batc hsizes, our implementation provides a consequent performance gain.

#### Figure: {#gpu-bench}
![B=4](figures/runtimes/bench_cpu_4_4.png){width=40%}
:GPU benchmark, placeholder


These benchmarking results show that the proposed method leads to significant speedup (up to X10) in the computation of the gradient which would lead to drastic cost reduction for training a network.

# Implementation

Our probing algorithm is implemented in julia using BLAS on PU and CUBALS on GPU for the linear algebra computations. Code is on github.

# Conclusions

- Good performance for large image and/or batchsize
- Don't need many probing vector if batchsize lare enough
- Fairly suboptimal implementation leads to less impressive results on GPU but can be improved

# References