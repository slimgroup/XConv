
  

# README.md

  

Look at the [OLD_README.md](OLD_README.md) for installation tips.

## Peak Memory Usage

  

1. To sequentially compute the peak-memory usage for different probing-vector configurations, check [bash_scripts/bash_compute_vanillanet_peak_memory.sh](bash_scripts/bash_compute_vanillanet_peak_memory.sh)

  

2. To plot the memory-heatmap for different architectures, check [bash_scripts/bash_plot_memory_curves.sh](../bash_scripts/bash_plot_memory_curves.sh).

  
  

## Full Gradient

  
  

1) Generate a randomized `(image, label)` dataset for every image dimension. Run [bash_scripts/bash_create_img_and_label_pairs.sh](bash_scripts/bash_create_img_and_label_pairs.sh).

  

  

  

2) To only compute the full gradients of a model, check [bash_scripts/bash_comp_full_gradient.sh](bash_scripts/bash_comp_full_gradient.sh)

  
  

## Mini-Batch Gradients and Average Gradient Error

  

  

  

1) Generate a randomized `(image, label)` dataset for every image dimension. Run [bash_scripts/bash_create_img_and_label_pairs.sh](bash_scripts/bash_create_img_and_label_pairs.sh).

  

  

  

2) To compute the mini-batch gradients and the average gradient error, check [bash_scripts/bash_comp_mini_batch_gradient_avg_grad_err.sh](bash_scripts/bash_comp_mini_batch_gradient_avg_grad_err.sh).

  

  
  

## Average Gradient Error

  

  

  

  

  

To reproduce the results, computing the average gradient error for different architectures, follow the steps:

  

  

  

  

  

1. Sequentially, compute the **full_gradients**, **mini-batch gradients** and the **average-gradient errors**, follow [bash_scripts/bash_seq_comp_avg_grad_err.sh]( bash_scripts/bash_seq_comp_avg_grad_err.sh).

  

  

  

  

  

2. To plot the error-curves for two different probing-vector configurations, follow [bash_scripts/bash_plot_err_curves_with_std_vanillanet.sh](../bash_scripts/bash_plot_err_curves_with_std_vanillanet.sh)

  

  

  

  

  

3. To plot and analyze how peak-memory changes with different probing-vector configurations for a fixed image-dimension, check [bash_scripts/bash_plot_vanillanet_memory_curves.sh](bash_scripts/bash_plot_vanillanet_memory_curves.sh)