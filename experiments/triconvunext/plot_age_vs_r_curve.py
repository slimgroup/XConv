import os
import re
import sys
import yaml
import argparse
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

"""
    Plot the average gradient error curves using mean and standard deviation across all probing vectors.

    Usage:
        sh bash_scripts/bash_plot_age_vs_r_curve.sh

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot average gradient error curves using mean and standard deviation across all probing vectors.'
    )
    parser.add_argument(
        '--err_dir', 
        type=str, 
        required=True, 
        help='Directory containing the average gradient errors.'
    )
    parser.add_argument(
        '--plot_dir',
        type = str,
        required= True,
        help = "Directory to save the plots"
    )
    parser.add_argument(
        '--model_name',
        type = str,
        required= True,
        help = "Name of the model."
    )
    parser.add_argument(
        '--xconv_varn',
        type = str,
        required= True,
        help = "XConv Variation: (XConv/Adaptive)."
    )
    parser.add_argument(
        '--probing_vectors',
        nargs='+', 
        required=True, 
        help='Probing vector list (e.g., "base", "16")'
    )
    parser.add_argument(
        '--img_dims',
        nargs='+', 
        type=int,
        required=False, 
        help='Image Dimensions List (e.g., 128, 256)'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        required=True,
        help='Configuration file for the plot.'
    )

    args = parser.parse_args()

    return args

def return_run_num(path):
    # Extract the run number from the path string
    match = re.search(r'run_num_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        return None
    


def compute_probing_vector_stats(
    img_dims, 
    err_dir, 
    img_dims_batch_sizes, 
    n_runs=10
):

    # shape (n_dims, n_runs)
    errors_by_dim = np.zeros((len(img_dims), n_runs))

    # Fill the errors_by_dim array with data
    for i, img_dim in enumerate(img_dims):
        batch_size = img_dims_batch_sizes[img_dim]
        for j, grad_file in enumerate(glob.glob(f"{err_dir}/img_{img_dim}*_batch*{batch_size}*")):
            err_file_path = os.path.join(grad_file, "avg_grad_err.pkl")
            run_num = return_run_num(grad_file)
            
            with open(err_file_path, "rb") as f:
                err = pickle.load(f)
            print("Storing error from file:", err_file_path, "run_num:", run_num, "err:", err)
            errors_by_dim[i, run_num - 1] = err

    # Compute mean and std deviation across runs for each image dimension
    mean_err = errors_by_dim.mean(axis=1)                  # (n_dims,)
    std_err  = errors_by_dim.std(axis=1)                   # unbiased std

    return mean_err, std_err

def format_conv_name(name: str) -> str:
    """
    'xconv'            -> 'XConv'
    'adaptive_xconv'   -> 'AdaptiveXConv'
    'conv'             -> 'Conv'
    """
    out = []
    for token in name.strip().lower().split('_'):
        if token.endswith('conv'):
            head = token[:-4]              # part before 'conv'
            head_fmt = head.capitalize() if head else ''
            out.append(head_fmt + 'Conv')  # force 'Conv' casing
        else:
            out.append(token.capitalize())
    return ''.join(out)


def clean_name(model_name):

    cleaned_name = re.sub(r'[^a-zA-Z]', '', model_name)
    cleaned_name = cleaned_name.capitalize()

    return cleaned_name

def plot_err_curve_with_std_dev(
    probing_vectors: list,
    pv_mean_errs,
    pv_std_errs,
    model_plot_name,
    plot_dir,
    probing_vector_img_dims_batch_sizes,
    img_dims
):
    """
        Plot Probing Vector vs Average Gradient Error plot with Standarad Deviation.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Baseline horizontal line (Conv)
    base_mean = float(np.asarray(pv_mean_errs['base']).squeeze())
    ax.axhline(
        y = base_mean,
        linestyle = '--',
        linewidth = 1.5,
        color="black",
        label = f"Conv. "
    )

    # XConv points: use numeric r and sort.
    r_vals = sorted(int(pv) for pv in probing_vectors if pv != 'base')
    y = np.array(
        [float(np.asarray(pv_mean_errs[str(r)]).squeeze()) for r in r_vals],
        dtype=float
    )
    y_err = np.array(
        [float(np.asarray(pv_std_errs[str(r)]).squeeze()) for r in r_vals],
        dtype=float
    )

    # Convert to 1-dim if not.
    y = y.squeeze()
    y_err = y_err.squeeze()

    # Mean curve +- markers
    ax.plot(
        r_vals,
        y,
        "-o", 
        lw=2, 
        ms=6, 
        label="XConv (mean)"
    )


    # Shaded band
    ax.fill_between(
        x = r_vals, 
        y1 = y - y_err, 
        y2 = y + y_err, 
        alpha=0.2, 
        label=r"XConv ($\pm \sigma$)"
    )

    # Batch-size annotations
    c1 = '#d62728'
    img_dim = img_dims[0]

    if 'base' in probing_vector_img_dims_batch_sizes:
        base_bs = probing_vector_img_dims_batch_sizes['base'][img_dim]
        ax.annotate(
            str(base_bs),
            xy=(r_vals[0], base_mean),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=9,
            color='black',
        )

    for r in r_vals:
        pv = str(r)
        bs = probing_vector_img_dims_batch_sizes[pv][img_dim]
        err_pv = float(np.asarray(pv_mean_errs[pv]).squeeze())
        ax.annotate(
            str(bs),
            xy=(r, err_pv),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=9,
            color=c1,
        )

    ax.set_xlabel("Probing vector $(r)$")
    ax.set_ylabel("Average Gradient Error (AGE)")
    ax.set_title(f"AGE vs Probing Vector ({model_plot_name})")

    ax.set_xticks(r_vals)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, loc="best")

    # Set log-scale
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    plot_path = os.path.join(
        plot_dir, 
        f"avg_grad_err_vs_probing_vectors.png"
    )
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig.tight_layout()
    print("Saving plot to:", plot_path)
    fig.savefig(
        plot_path, 
        dpi=300, 
        format="png",
        bbox_inches="tight"
    )

    return 

def main(args):

    probing_vectors = args.probing_vectors
    model_name = args.model_name
    xconv_varn = args.xconv_varn
    config_file = args.config_file
    err_dir = args.err_dir
    plot_dir = args.plot_dir

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Step 2: Derive the EXP key BEFORE formatting
    exp_key = f"{model_name}_{xconv_varn}"   # the literal key in YAML

    # Validate config structure
    if exp_key not in config:
        raise KeyError(f"Experiment '{exp_key}' not found in {config_file}")

    # Step 4: Now index the SAME exp_key (not exp_name)
    # Otherwise the second lookup breaks.
    exp_cfg = config[exp_key]

    img_dims = exp_cfg["img_dims"]
    model_plot_name = exp_cfg["model_plot_name"]
    probing_vector_img_dims_batch_sizes = exp_cfg["probing_vector_img_dims_batch_sizes_dict"]

    mean_errs = {}
    std_errs = {}

    for probing_vector in probing_vectors:
        pv_dir = os.path.join(err_dir, f"probing_vector_{probing_vector}")
        mean_err_pv, std_err_pv = compute_probing_vector_stats(
            img_dims = img_dims, 
            err_dir = pv_dir, 
            img_dims_batch_sizes = probing_vector_img_dims_batch_sizes[probing_vector]
        )
        mean_errs[probing_vector] = mean_err_pv
        std_errs[probing_vector] = std_err_pv

    xconv_varn = format_conv_name(xconv_varn)

    plot_err_curve_with_std_dev(
        probing_vectors = probing_vectors,
        pv_mean_errs = mean_errs,
        pv_std_errs = std_errs,
        model_plot_name = model_plot_name,
        plot_dir = plot_dir,
        probing_vector_img_dims_batch_sizes = probing_vector_img_dims_batch_sizes,
        img_dims = img_dims
    )
    

    
if __name__ == "__main__":
    args = parse_args()
    main(args)