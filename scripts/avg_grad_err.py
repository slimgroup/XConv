import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import argparse
from torchvision import models

"""
Usage:
    python3 compute_avg_gradient_error.py --img_size 

Arguments:
    --img_size      (required) Image Dimension
    --subset_size [1024] Subset Size selected for this experiment.
    --batch_size  (required) Batch-size.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Compute Average gradient error from the given mini-batch and full gradients.')
    parser.add_argument('--full_grads_path', type=str, required=True, help='Path to the pickle file containing the full gradients.')
    parser.add_argument('--mini_grads_path', type=str, required=True, help='Path to the pickle file containing the per-batch gradients.')
    parser.add_argument('--grad_dict_dir', type=str, required=True, help='Path to the directory containing the dict containing gradients.')

    args = parser.parse_args()

    return args

def compute_avg_grad_err(
    mini_batch_grads, 
    full_model_grads, 
    model_layer = None, 
    conv_param_names=None
):
    """
    Computes the average L2 error between mini-batch gradients and the full dataset gradient
    for a specific model layer.

    Args:
        mini_batch_grads (dict): Dictionary mapping batch indices to parameter gradient dicts.
                                 Format: mini_batch_grads[batch_idx][param_name] = gradient_tensor
        full_model_grads (dict): Dictionary mapping parameter names to full-dataset gradients.
                                 Format: full_model_grads[param_name] = gradient_tensor
        model_layer (str): The name of the layer for which to compute the gradient error
                           (e.g., 'conv1.weight').

    Returns:
        float: Average L2 error (Euclidean distance) between each mini-batch gradient
               and the full-dataset gradient for the specified layer.
    """

    batch_errors = []

    # If no specific layer is provided, compute error across all convolutional layers.
    if model_layer is None:
        for batch_idx in mini_batch_grads:

            # The mini-batch gradient for the current batch.
            batch_grad = mini_batch_grads[batch_idx]
            
            # Store the l2 errors for each convolutional layer.
            l2_layer_errs_list = []
            # p_cnt = 0
            
            for layer in conv_param_names:

                print("Computing gradient error for {}".format(layer))

                # Paper Eq. (9): squared L2 error per conv-weight tensor,
                # accumulated in fp32 (matches pyxconv/radcompare/age.py). Summing these
                # squares equals ||concat(g - g^(b))||_2^2 over the conv weights.
                curr_layer_l2_err = (
                    full_model_grads[layer].float() - batch_grad[layer].float()
                ).pow(2).sum()
                l2_layer_errs_list.append(curr_layer_l2_err.item())

                # Normalize by number of parameters of the Convolution layers.
                # p_cnt += (batch_grad[layer].numel())

            # batch_errors.append(np.sum(l2_layer_errs_list)/p_cnt)

            # Sum of per-layer squared L2 errors = squared L2 over all conv weights.
            batch_errors.append(np.sum(l2_layer_errs_list))
    else:

        for batch_idx in mini_batch_grads:
            batch_grad = mini_batch_grads[batch_idx][model_layer]
            full_grad = full_model_grads[model_layer]

            l2_error = torch.linalg.vector_norm(full_grad - batch_grad, ord=2)
            batch_errors.append(l2_error.item())

    avg_error = np.mean(batch_errors)

    return avg_error

def get_conv_param_names(model):
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    conv_param_names = set()

    for module_name, module in model.named_modules():
        if isinstance(module, conv_types):
            for param_name, _ in module.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                conv_param_names.add(full_param_name)

    return conv_param_names

def main(args):    

    mini_grads_path = args.mini_grads_path
    full_grads_path = args.full_grads_path
    grad_dict_dir = args.grad_dict_dir

    with open(mini_grads_path, "rb") as f:
        mini_batch_grad_param_dict = pickle.load(f)

    with open(full_grads_path, "rb") as f:
        full_grad_param_dict = pickle.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, "squeezenet1_0")().to(device)

    conv_param_names = get_conv_param_names(model)

    # Compute the average gradient errors.
    avg_err = compute_avg_grad_err(
        mini_batch_grads= mini_batch_grad_param_dict, 
        full_model_grads=full_grad_param_dict,
        conv_param_names=conv_param_names
    )
    print(f"Avg L2 gradient error: {avg_err:.6f}")

    avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
    with open(avg_grad_err_file_path, "wb") as f:
        pickle.dump(avg_err, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()
    main(args)