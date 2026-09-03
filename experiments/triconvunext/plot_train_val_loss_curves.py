import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

"""
    Plot an overlayed Conv vs XConv train and validation loss curves from the log CSV file.

    Usage:
        sh bash_scripts/bash_plot_train_val_loss_curves.sh
"""

def plot_train_val_loss_curves(
    base_log_df: pd.DataFrame,
    comp_log_df: pd.DataFrame,
    output_dir: str,
    model_name: str,
    cmp_split: str
):
    """
        Plot the train/validation loss curves for the base and comparison models.
        Args:
            base_log_df: pd.DataFrame, the log dataframe of the base model.
            comp_log_df: pd.DataFrame, the log dataframe of the comparison model.
            output_dir: str, the directory to save the plot.
            model_name: str, the name of the model.
            cmp_split: str, the split to compare the loss curves.
        Returns:
            None
    """
    if cmp_split == "train":
        base_loss = base_log_df['loss']
        comp_loss = comp_log_df['loss']
    elif cmp_split == "val":
        base_loss = base_log_df['val_loss']
        comp_loss = comp_log_df['val_loss']

    epoch = base_log_df['epoch']
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    # Plot the standard convolution loss plot.
    ax.plot(
        epoch, 
        base_loss, 
        label='Conv', 
        color='blue'
    )

    # Plot the XConv [r = 1024] loss plot.
    ax.plot(
        epoch, 
        comp_loss, 
        label='XConv [r = 1024]', 
        color='red'
    )

    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    ax.set_title(f'{model_name} {cmp_split.capitalize()} Loss Curves')

    ax.legend(
        loc='upper right', 
        frameon=True
    )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
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

    plot_save_path = os.path.join(output_dir, f'{model_name}_{cmp_split}_loss_curves.png')
    print("Saving {}".format(plot_save_path))
    plt.savefig(
        plot_save_path,
        dpi = 300,
        format="png",
        bbox_inches = 'tight'
    )
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_log_csv_file", 
        type=str, 
        required=True,
        help="Path to the log CSV file of the base model."
    )
    parser.add_argument(
        "--comp_log_csv_file", 
        type=str, 
        required=True,
        help="Path to the log CSV file of the model of different probing vector."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Directory to save the loss curves."
    )
    parser.add_argument(
        "--cmp_split", 
        type=str, 
        required=True,
        help="Split to compare the loss curves.",
        choices=["train", "val"]
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Name of the model."
    )
    return parser.parse_args()

def main(args: argparse.Namespace):

    base_log_csv_file = args.base_log_csv_file
    comp_log_csv_file = args.comp_log_csv_file
    cmp_split = args.cmp_split

    model_name = args.model_name

    assert os.path.exists(base_log_csv_file), f"Log CSV file {base_log_csv_file} does not exist."
    assert os.path.exists(comp_log_csv_file), f"Log CSV file {comp_log_csv_file} does not exist."

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_log_df = pd.read_csv(base_log_csv_file)
    comp_log_df = pd.read_csv(comp_log_csv_file)

    plot_train_val_loss_curves(
        base_log_df=base_log_df, 
        comp_log_df=comp_log_df, 
        output_dir=output_dir, 
        model_name=model_name,
        cmp_split=cmp_split
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)