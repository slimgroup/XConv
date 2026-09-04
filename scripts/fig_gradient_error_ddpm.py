import os
import re
import sys
import yaml
import argparse
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, LogLocator

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot average gradient error curves using mean and standard deviation for two different cases.'
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

sips_unet_probing_vector_img_dims_batch_sizes = {
    "base": {
        28: 660,
        64: 200,
        128: 50,
        256: 12
    },
    "16": {
        28: 660,
        64: 275,
        128: 68,
        256: 16
    },
    "32": {
        28: 660,
        64: 273,
        128: 68,
        256: 16
    },
    "64": {
        28: 660,
        64: 270,
        128: 66,
        256: 16
    },
    "128": {
        28: 660,
        64: 263,
        128: 67,
        256: 16
    },
    "256": {
        28: 660,
        64: 250,
        128: 66,
        256: 16
    },
    "512": {
        28: 660,
    }
}

vanillanet_probing_vector_img_dims_batch_sizes = {
    "base": {
        64: 1584,
        128: 349,
        256: 81,   
        512: 19
    },
    "64": {
        64: 1055,
        128: 386,
        256: 106,
        512: 26
    },
    "32": {
        64: 1574,
        128: 476,
        256: 120,
        512: 30
    },
    "16": {
        64: 1998,
        128: 527,
        256: 129,
        512: 31
    },
    "4": {
        64: 2477,
        128: 573,
        256: 135,
        512: 32
    }
}

score_sde_unet_probing_vector_img_dims_batch_sizes = {
    "base": {
        1024: 3
    },
    "16": {
        1024: 6
    },
    "32": {
        1024: 6
    },
    "64": {
        1024: 6
    },
    "128": {
        1024: 6
    },
    "256": {
        1024: 6
    }
}

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
    name = name.split("_bf")[0]
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


CONV_COLOR = '#1f77b4'
XCONV_COLOR = '#d62728'


def load_experiment_config(config_file, model_name, xconv_varn):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    exp_key = f"{model_name}_{xconv_varn}"
    if exp_key not in config:
        raise KeyError(f"Experiment '{exp_key}' not found in {config_file}")

    exp_cfg = config[exp_key]
    return (
        exp_cfg["img_dims"],
        exp_cfg["model_plot_name"],
        exp_cfg["probing_vector_img_dims_batch_sizes_dict"],
    )


def load_mean_std_for_probing_vectors(
    err_dir,
    img_dims,
    probing_vectors,
    probing_vector_img_dims_batch_sizes,
    n_runs=10,
):
    mean_errs = {}
    std_errs = {}
    for probing_vector in probing_vectors:
        pv_dir = os.path.join(err_dir, f"probing_vector_{probing_vector}")
        mean_err_pv, std_err_pv = compute_probing_vector_stats(
            img_dims=img_dims,
            err_dir=pv_dir,
            img_dims_batch_sizes=probing_vector_img_dims_batch_sizes[probing_vector],
            n_runs=n_runs,
        )
        mean_errs[probing_vector] = mean_err_pv
        std_errs[probing_vector] = std_err_pv
    return mean_errs, std_errs


def annotate_batch_sizes(ax, img_dims, mean_errs, bs_pv1, bs_pv2, probing_vector1, probing_vector2):
    errs_pv1 = mean_errs[probing_vector1]
    errs_pv2 = mean_errs[probing_vector2]
    c1, c2 = CONV_COLOR, XCONV_COLOR

    for img_dim, err_pv1, err_pv2 in zip(img_dims, errs_pv1, errs_pv2):
        if err_pv1 >= err_pv2:
            ax.annotate(
                str(bs_pv1[img_dim]),
                xy=(img_dim, err_pv1),
                textcoords="offset points",
                xytext=(-1, +12),
                ha='center',
                fontsize=9,
                color=c1,
            )
            ax.annotate(
                str(bs_pv2[img_dim]),
                xy=(img_dim, err_pv2),
                textcoords="offset points",
                xytext=(+8, -15),
                ha='center',
                fontsize=9,
                color=c2,
            )
        else:
            ax.annotate(
                str(bs_pv1[img_dim]),
                xy=(img_dim, err_pv1),
                textcoords="offset points",
                xytext=(-1, -15),
                ha='center',
                fontsize=9,
                color=c1,
            )
            ax.annotate(
                str(bs_pv2[img_dim]),
                xy=(img_dim, err_pv2),
                textcoords="offset points",
                xytext=(+8, +12),
                ha='center',
                fontsize=9,
                color=c2,
            )


def plot_comparison_panel(
    ax,
    img_dims,
    mean_errs,
    std_errs,
    probing_vector_img_dims_batch_sizes,
    xconv_varn_label,
    model_plot_name,
    xconv_pv,
    probing_vector1="base",
    show_ylabel=True,
    panel_label=None,
):
    probing_vector2 = xconv_pv if isinstance(xconv_pv, str) else str(xconv_pv)

    ax.plot(
        img_dims,
        mean_errs[probing_vector1],
        '-o',
        lw=2,
        ms=6,
        color=CONV_COLOR,
        label='Mean gradient error for Conv',
        zorder=3,
    )
    ax.fill_between(
        img_dims,
        mean_errs[probing_vector1] - std_errs[probing_vector1],
        mean_errs[probing_vector1] + std_errs[probing_vector1],
        color=CONV_COLOR,
        alpha=0.2,
        label=r'Spread of gradient error for Conv ($\pm \sigma$)',
        zorder=2,
    )
    ax.plot(
        img_dims,
        mean_errs[probing_vector2],
        '-o',
        lw=2,
        ms=6,
        color=XCONV_COLOR,
        label='Mean gradient error for {}'.format(xconv_varn_label),
        zorder=3,
    )
    ax.fill_between(
        img_dims,
        mean_errs[probing_vector2] - std_errs[probing_vector2],
        mean_errs[probing_vector2] + std_errs[probing_vector2],
        color=XCONV_COLOR,
        alpha=0.2,
        label=r'Spread of gradient error for {} ($\pm \sigma$)'.format(xconv_varn_label),
        zorder=2,
    )

    annotate_batch_sizes(
        ax,
        img_dims,
        mean_errs,
        probing_vector_img_dims_batch_sizes[probing_vector1],
        probing_vector_img_dims_batch_sizes[probing_vector2],
        probing_vector1,
        probing_vector2,
    )

    title = f'Average Gradient Error for {model_plot_name} ($r = {probing_vector2}$)'
    if panel_label:
        title = f'{panel_label} {title}'
    ax.set_title(title)
    ax.set_xlabel('Image dimension')
    if show_ylabel:
        ax.set_ylabel('Average Gradient Error')
    ax.set_yscale('log', base=10)
    ax.grid(True, alpha=0.3)


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


    _, ax = plt.subplots(figsize=(8, 5))

    probing_vector1 = "base"
    probing_vector2 = probing_vectors[1]

    xconv_varn = format_conv_name(xconv_varn)

    # Plot for probing vector 1
    # It is the true gradient.
    ax.plot(
        img_dims, 
        mean_errs[probing_vector1], 
        '-o', 
        lw=2, 
        ms=6, 
        color='#1f77b4',
        label='Mean gradient error for Conv',
        zorder=3
    )
    ax.fill_between(
        img_dims, 
        mean_errs[probing_vector1] - std_errs[probing_vector1], 
        mean_errs[probing_vector1] + std_errs[probing_vector1],
        color='#1f77b4',
        alpha=0.2, 
        label=r'Spread of gradient error for Conv ($\pm \sigma$)',
        zorder = 2
    )

    # Plot for probing vector 2
    ax.plot(
        img_dims, 
        mean_errs[probing_vector2], 
        '-o', 
        lw=2, 
        ms=6, 
        color='#d62728',
        label='Mean gradient error for {}'.format(xconv_varn),
        zorder=3
    )
    ax.fill_between(
        img_dims, 
        mean_errs[probing_vector2] - std_errs[probing_vector2], 
        mean_errs[probing_vector2] + std_errs[probing_vector2],
        color='#d62728',
        alpha=0.2, 
        label=r'Spread of gradient error for {} ($\pm \sigma$)'.format(xconv_varn),
        zorder = 2
    )

    # Add the batch-sizes as annotations over the points
    errs_pv1 = mean_errs[probing_vector1]

    bs_pv1 = probing_vector_img_dims_batch_sizes[probing_vector1]
    bs_pv2  = probing_vector_img_dims_batch_sizes[probing_vector2]
    errs_pv2  = mean_errs[probing_vector2]

    c1, c2 = '#1f77b4', '#d62728'

    for img_dim, err_pv1, err_pv2 in zip(img_dims, errs_pv1, errs_pv2):
        # decide which curve is higher at this x
        if err_pv1 >= err_pv2:
            # pv1 above, pv2 below
            ax.annotate(
                str(bs_pv1[img_dim]), 
                xy=(img_dim, err_pv1),
                textcoords="offset points", 
                xytext=(-1, +12),
                ha='center', 
                fontsize=9, 
                color=c1
            )
            ax.annotate(
                str(bs_pv2[img_dim]), 
                xy=(img_dim, err_pv2),
                textcoords="offset points", 
                xytext=(+8, -15),
                ha='center', 
                fontsize=9, 
                color=c2
            )
        else:
            # pv2 above, pv1 below
            ax.annotate(
                str(bs_pv1[img_dim]), 
                xy=(img_dim, err_pv1),
                textcoords="offset points", 
                xytext=(-1, -15),
                ha='center', 
                fontsize=9, 
                color=c1
            )
            ax.annotate(
                str(bs_pv2[img_dim]), 
                xy=(img_dim, err_pv2),
                textcoords="offset points", 
                xytext=(+8, +12),
                ha='center', 
                fontsize=9, 
                color=c2
            )

    # Formatting
    ax.set_title(f'Average Gradient Error for {model_plot_name} (r = {probing_vector})')
    ax.set_xlabel('Image dimension')

    # Set log-scale
    # ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_ylabel('Average Gradient Error')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc='best')

    plot_path = os.path.join(plot_dir, f"error_plot_probing_vectors_{probing_vector1}_{probing_vector2}.png")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.tight_layout()
    print("Saving plot to:", plot_path)
    plt.savefig(
        plot_path, 
        dpi=300, 
        format="png",
        bbox_inches="tight"
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)