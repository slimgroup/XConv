[![DOI](https://zenodo.org/badge/346371507.svg)](https://zenodo.org/badge/latestdoi/346371507)

# XConv

Code for

> **XConv: Low-memory stochastic backpropagation for convolutional layers.**
> Anirudh Thatipelli, Jeffrey J. Sam, Mathias Louboutin, Ali Siahkoohi, Rongrong Wang and
> Felix J. Herrmann. Transactions on Machine Learning Research, 2026.
> https://openreview.net/forum?id=ajv7wvEvnh

## Overview

Training a convolutional network holds every intermediate activation until the backward pass
needs it, and on high-resolution or volumetric data that is what exhausts the device long before
the arithmetic does. XConv keeps a compressed projection of each activation instead of the
activation itself, and recovers the filter gradient from it by multi-channel randomized trace
estimation. The number of probing vectors sets the trade: more of them cost memory and buy
gradient accuracy.

It is a near drop-in replacement. Backpropagation stays standard, the architecture is unchanged,
and an existing network is converted in one call.

| | |
|---|---|
| Julia | overloads [NNlib](https://github.com/FluxML/NNlib.jl)'s `∇conv_filter`, so an existing model needs no edit |
| PyTorch | `Xconv2D` and `Xconv3D` layers, and `convert_net` to swap them into a model in place |

Both support 2D and 3D convolutions.

## Installation

Julia:

```julia
]dev https://github.com/slimgroup/XConv
```

PyTorch:

```bash
pip install git+https://github.com/slimgroup/XConv
```

To also get the experiments that reproduce the paper:

```bash
git clone https://github.com/slimgroup/XConv
cd XConv
pip install -e .
```

This pulls the default `torch`. For a build matched to your system, follow
[the PyTorch installation guide](https://pytorch.org/get-started/locally/). Reading device memory
through NVML, which the peak-memory measurements report, needs the optional extra:
`pip install -e ".[memory]"`.

## Using it

```python
from pyxconv import convert_net

convert_net(model, ps=16)   # every convolution now stores a rank-16 projection
```

`ps` is the number of probing vectors. `adaptive_convert_net` converts only those layers whose
activation is large enough to be worth compressing, which is the better choice for networks that
end in many small feature maps.

## Reproducing the paper

`scripts/` holds the method figures and the measurements behind them; `experiments/` holds one
directory per downstream task. A `fig_` or `tab_` prefix means the script renders a result;
anything else produces what one of them reads.

## Tests

```bash
pytest tests/
```

CPU-only and fast. They assert properties rather than outputs: the filter-gradient estimate is
averaged over many probe draws and compared against the exact gradient, in 2D and 3D, for each
probing distribution and each boundary.

## Authors

Developed at the Georgia Institute of Technology by the ML4Seismic Lab.

- Mathias Louboutin — <mlouboutin3@gatech.edu>
- Ali Siahkoohi — <alisk@ucf.edu>
- Anirudh Thatipelli — <anirudh.thatipelli@ucf.edu>
- Jeffrey J. Sam — <jeffrey.jj.sam@gmail.com>

## Acknowledgment

This software was developed and tested on GPUs thanks to an NVIDIA Academic Hardware Grant.

## License

MIT — see [LICENSE](LICENSE).
