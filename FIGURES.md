# Reproducing the figures and tables

Every figure and table in

> **XConv: Low-memory stochastic backpropagation for convolutional layers.**
> Transactions on Machine Learning Research, 2026. https://openreview.net/forum?id=ajv7wvEvnh

and the script that produces it. `scripts/` holds the method figures and the measurements behind
them; `experiments/` holds one directory per downstream task. A `fig_` or `tab_` prefix means the
script renders a result; anything else produces what one of them reads.

## Method figures

| figure | shows | script | reads |
|---|---|---|---|
| 3a | per-weight gradient estimates, four convolution layers | `scripts/fig_gradient_estimates.py` | — |
| 3b | gradient standard deviation against batch size | `scripts/fig_gradient_variance.py` | `scripts/gradient_variance_sweep.py` |
| 4, 26 | SqueezeNet gradient error against the probing-vector count | `scripts/fig_gradient_error.py` | `configs/probing_vectors/` |
| 5, 27 | U-Net gradient error against the probing-vector count | `scripts/fig_gradient_error.py` | `configs/rad_vs_xconv_unet_age.json` |
| 6, 28 | VanillaNet gradient error against the probing-vector count | `scripts/fig_gradient_error.py` | `experiments/vanillanet/pv_configs/` |
| 7 | layer-by-layer memory, four networks | `scripts/fig_layer_memory.py` | — |
| 8, 32 | SqueezeNet peak memory | `scripts/peak_memory_squeezenet.py` | `configs/squeezenet_peak_memory.json` |
| 9, 34 | U-Net peak memory | `scripts/peak_memory_unet.py` | `configs/rad_vs_xconv_unet_age.json` |
| 10, 33 | VanillaNet peak memory | `experiments/vanillanet/compute_vanillanet_peak_memory.py` | — |
| 11, 29 | CPU runtime benchmark | `scripts/fig_runtime_cpu.jl` | — |
| 12, 30 | GPU runtime benchmark | `scripts/fig_runtime_gpu.py` | `scripts/runtime_gpu_bench.py` |
| 37 | DDPM U-Net gradient error against image dimension | `scripts/fig_gradient_error_ddpm.py` | `configs/err_plots/sips_unet_xconv.yaml` |

## Application figures

| figure | shows | script | reads |
|---|---|---|---|
| 13 | CIFAR-10 training at equal memory | `scripts/fig_cifar_training.py` | `scripts/cifar10_train.py` |
| 14 | facies classification, gradient error | `experiments/facies_classification/plot_avg_grad_err_vs_r.py` | `experiments/facies_classification/err_plot_configs/` |
| 15 | MNIST generated samples | `experiments/sips/scripts/mnist_example.py` | `experiments/sips/configs/` |
| 16 | MNIST FID against the probing-vector count | `experiments/sips/scripts/plot_fid.py` | `experiments/sips/scripts/compute_fid.py` |
| 17 | CIFAR-10 gradient error | `experiments/cifar10-generation/scripts/plot_avg_grad_err_vs_r.py` | `experiments/cifar10-generation/configs/` |
| 18 | CIFAR-10 generated samples | `experiments/cifar10-generation/scripts/cifar10.py` | `experiments/cifar10-generation/configs/cifar10_example.json` |
| 20 | seismic gradient error | `experiments/genparihaka/scripts/plot_age_vs_r_curve.py` | `experiments/genparihaka/configs/` |
| 21 | deep image prior, super-resolution | `experiments/deep-image-prior/scripts/super_resolution_table1.py` | `experiments/deep-image-prior/data/sr/` |
| 22 | deep image prior, inpainting | `experiments/deep-image-prior/scripts/inpainting.py` | `experiments/deep-image-prior/data/inpainting/` |
| 23a | TriConvUNeXt gradient error | `experiments/triconvunext/plot_age_vs_r_curve.py` | `experiments/triconvunext/err_plot_configs/` |
| 23b | TriConvUNeXt peak memory | `experiments/sips/scripts/plot_memory_curves.py` | `experiments/triconvunext/compute_peak_memory.py` |
| 23c | TriConvUNeXt training curves | `experiments/triconvunext/plot_train_val_loss_curves.py` | `experiments/triconvunext/train.py` |
| 24 | gland segmentation, qualitative | `experiments/triconvunext/val.py` | `experiments/triconvunext/save_mask_black_white.py` |
| 25 | spleen segmentation, axial slices | `experiments/3d-segmentation/visualize.py` | `experiments/3d-segmentation/run.py` |
| 31a | super-resolution PSNR against peak memory | `experiments/deep-image-prior/scripts/plot_super_resolution_psnr_memory_curve.py` | — |
| 31b | inpainting peak memory | `experiments/deep-image-prior/scripts/plot_inpainting_peak_memory_curve.py` | `experiments/deep-image-prior/scripts/compute_inpainting_peak_memory.py` |
| 35 | MNIST accuracy against epoch | `scripts/fig_mnist_accuracy.py` | `scripts/mnist_train.py` |
| 36 | U-Net DDPM training curves | `experiments/sips/scripts/plot_multiple_sips_plots.py` | — |

## Tables

| table | shows | script |
|---|---|---|
| 1 | MNIST accuracy, Julia | `scripts/tab_mnist_accuracy.jl` |
| 2 | facies classification metrics | `experiments/facies_classification/patch_test.py` |
| 3 | super-resolution and inpainting PSNR | `experiments/deep-image-prior/scripts/compute_psnr_dip_outputs.py` |
| 4 | gland segmentation Dice and accuracy | `experiments/triconvunext/val.py` |
| 5 | spleen segmentation Dice and peak memory | `experiments/3d-segmentation/run.py` |
| 6 | VanillaNet channel counts per stage | `experiments/vanillanet/comp_vanillanet_layer_histogram.py` |

## Assembled by hand

These have no generating script: Figure 1, the schematic; Tables 7, 8 and 9, the architecture
listings; Figure 19, the seismic sample montages, whose tiles come from
`experiments/genparihaka/genparihaka/utils/plotting.py`; and the unprocessed reference panels of
Figures 21, 22 and 24, which are crops of files under each experiment's `data/`.

## Data

The summary tables and the gradient-error records behind the published runs:
https://www.dropbox.com/scl/fo/umrpgs2h9ns9pjrarhib6/ABzQlq30EEXnXCGF0bhP1FU?rlkey=5m96adh5k9ss6gpjq2gtkfkbe&dl=0
