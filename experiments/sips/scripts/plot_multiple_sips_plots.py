import os
import numpy as np
from matplotlib.ticker import ScalarFormatter
import argparse
import torch
from pathlib import Path
import matplotlib.pyplot as plt

"""
Plot the training/validation loss curves from Conv and XConv models.

Usage:
    sh bash_scripts/plot_multiple_sips_plots.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Plot the training/validation loss curves from Conv and XConv models.")
    parser.add_argument(
        "--conv_checkpoint",
        type=str,
        required=True,
        help="Checkpoint file of Conv model.",
    )
    parser.add_argument(
        "--xconv_checkpoint",
        type=str,
        required=True,
        help="Checkpoint file of XConv model.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="sips_plots",
        help="Directory to",
    )
    parser.add_argument(
        "--probing_vector",
        type=str,
        required=True,
        help="Probing vector.",
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="train_val_loss.png",
        required=True,
        help="Name of the output plot file.",
    )
    parser.add_argument(
        "--plot_label",
        type=str,
        default="train",
        required=True,
        help="Label for the training/validation loss curves (e.g., 'train' or 'val').",
    )
    parser.add_argument(
        "--skip_first",
        type=int,
        default=0,
        required=True,
        help="Number of steps to skip from the beginning of the loss curves.",
    )

    return parser.parse_args()



def plot_xconv_vs_conv_loss_curves(
    conv_loss_iter: list[float], 
    xconv_loss_iter: list[float], 
    label: str,
    probing_vector: str,
    plots_dir: str,
    plot_name: str,
    skip_first: int = 10
) -> None:
    """
        Plot Conv. vs XConv. loss curves and save the plot in the specified directory.
    Args:
        conv_loss_iter: list of Conv. loss values
        xconv_loss_iter: list of XConv. loss values
        label: label for the plot (e.g., 'Training Loss' or 'Evaluation Loss')
        probing_vector: probing vector
        plots_dir: directory to save the plots
        plot_name: name of the output plot file
    Returns:
        None"""

    
    # Plot Conv.
    plt.figure(figsize=(7, 4.5))

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    if skip_first > 0:
        conv_loss_iter = conv_loss_iter[skip_first:]
        xconv_loss_iter = xconv_loss_iter[skip_first:]

    steps = np.arange(len(conv_loss_iter))

    plt.plot(
        steps, 
        conv_loss_iter, 
        lw = 1.5,
        label='Conv',
    )

    # Plot XConv.
    plt.plot(
        steps, 
        xconv_loss_iter, 
        lw = 1.5,
        label='XConv [r = {}]'.format(probing_vector),
    )

    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    plt.title(f"{label} Loss Comparison: Conv vs XConv")
        
    # Clean, paper-style grid
    plt.grid(
        True, 
        which="major", 
        linestyle="--", 
        alpha=0.25
    )
    plt.grid(
        False, 
        which="minor"
    )

    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, "{}.png".format(plot_name))
    print("Saving plot to:", plot_path)
    plt.savefig(
        plot_path,
        format="png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    return

def read_checkpoint_files(checkpoint: str) -> list[float]:
    """
    Read the loss values from the specified checkpoint file.
    Args:
        checkpoint: checkpoint file
    Returns:
        list of loss values
    """
    checkpoint = torch.load(
        checkpoint, 
        weights_only=False
    )
    
    train_loss_iter = checkpoint['train_obj']
    eval_loss_iter = checkpoint['val_obj']

    return train_loss_iter, eval_loss_iter

def main(args):
    conv_checkpoint = args.conv_checkpoint
    xconv_checkpoint = args.xconv_checkpoint
    probing_vector = args.probing_vector
    plots_dir = Path(args.plots_dir)
    plot_name = args.plot_name
    plot_label = args.plot_label
    skip_first = args.skip_first

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Read loss values from the specified checkpoint files.
    conv_train_loss_iter, conv_eval_loss_iter = read_checkpoint_files(conv_checkpoint)
    xconv_train_loss_iter, xconv_eval_loss_iter = read_checkpoint_files(xconv_checkpoint)

    # Plot both XConv and Conv on the same "training/evaluation loss" plot
    label = 'Training' if plot_label == 'train' else 'Evaluation'

    plot_name = "xconv_vs_conv_{}_loss_curves_probing_vector_{}".format('train' if plot_label == 'train' else 'eval', probing_vector)

    if plot_label == 'train':   
        plot_xconv_vs_conv_loss_curves(
            conv_loss_iter = conv_train_loss_iter, 
            xconv_loss_iter = xconv_train_loss_iter, 
            label = label,
            plots_dir = plots_dir,  
            plot_name = plot_name,
            probing_vector = probing_vector,
            skip_first = skip_first
        )
    else:
        plot_xconv_vs_conv_loss_curves(
            conv_loss_iter = conv_eval_loss_iter, 
            xconv_loss_iter = xconv_eval_loss_iter, 
            label = label,
            plots_dir = plots_dir,
            plot_name = plot_name,
            probing_vector = probing_vector,
            skip_first = skip_first
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)