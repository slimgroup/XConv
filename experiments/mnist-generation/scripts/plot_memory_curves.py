import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Plot memory curve showing how memory changes for different batch-sizes for the same image-dimension.

Usage:
    sh bash_scripts/bash_plot_memory_curves.sh

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot memory curve showing how memory changes for different batch-sizes for the same image-dimension.'
    )
    parser.add_argument(
        '--batch_sizes',
        nargs='+', 
        required=False, 
        help='Batch sizes (e.g., "5", "32")'
    )
    parser.add_argument(
        '--mem_dir', 
        type=str, 
        required=True, 
        help='Root directory for memory.'
    )
    parser.add_argument(
        '--mem_curves_save_dir', 
        type=str, 
        required=True,
        help='Path to save the final memory curve plot.'
    )
    parser.add_argument(
        '--exp_name', 
        type=str, 
        required=True, 
        help='Experiment Variation.'
    )
    parser.add_argument(
        '--plot_name', 
        type=str, 
        required=True, 
        help='Name of the model in the plot.'
    )
    parser.add_argument(
        '--img_dim', 
        type=int, 
        required=True, 
        help='Image Dimension'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        help='Model Name'
    )
    parser.add_argument(
        '--xconv_varn', 
        type=str, 
        required=True, 
        help='Type of Xconv variation (adaptive/xconv).'
    )
    parser.add_argument(
        "--x_axis_name",
        type=str,
        help = "Name of the x-axis label.",
        default = False
    )
    parser.add_argument(
        "--y_axis_name",
        type=str,
        help = "Name of the y-axis label.",
        default = False
    )
    parser.add_argument(
        "--is_poster",
        action = "store_true",
        help = "Use configuration needed for the poster."    
    )

    args = parser.parse_args()

    return args

def check_atleast_one_not_nan(mem_list):
    """
    Check if atleast one non-NaN value exist in the memory list.
    """
    atleast_one_not_nan = any(x == x for x in mem_list)

    return atleast_one_not_nan

def plot_memory_curves(
    df_heatmap, 
    probing_vectors, 
    batch_sizes, 
    img_dim, 
    plot_title,
    save_path = None,
    x_axis_name = "Probing vectors",
    y_axis_name = "Memory (GB)",
    is_poster = False,
    dpi=300
):
    """
    Plot memory vs probing vectors with semilogx x-axis.
    
    Args:
        df_heatmap (pd.DataFrame): DataFrame with rows=batch_sizes, columns=probing_vectors
        probing_vectors (list): list of probing vectors, e.g. ["base", 2, 4, 8, ...]
        batch_sizes (list): list of batch sizes
        img_dim (int): image dimension used in experiment
        save_path (str, optional): file path to save the figure
    """
    # preprocess probing vectors: keep only numeric values for x-axis
    x_vals = [int(pv) for pv in probing_vectors if pv != "base"]
    
    # prepare plot
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for bs in batch_sizes:
        vals_mb = df_heatmap.loc[bs].values.astype(float)
        vals_gb = vals_mb / 1024.0

        # filter out "base"
        if "base" in probing_vectors:
            base_idx = probing_vectors.index("base")
            base_val = vals_gb[base_idx]
            vals_gb = np.delete(vals_gb, base_idx)

        # Don't add BS to legend for the poster.
        legend_label = f'BS={bs}' if not is_poster else None

        
        # Add to the legend only if values are not NaN
        if check_atleast_one_not_nan(vals_gb):
            # Log-scaling on both axes
            if len(batch_sizes) > 1:
                ax.loglog(
                    x_vals,
                    vals_gb,
                    '-o',
                    lw=2,
                    ms=6,
                    label=legend_label
                )
            else:
                ax.loglog(
                    x_vals,
                    vals_gb,
                    '-o',
                    lw=2,
                    ms=6,
                )

        # plot base line
        baseline_label = "Standard method" if is_poster else None

        if base_val is not None and np.isfinite(base_val):
            ax.axhline(
                y=base_val,
                linestyle='--',
                color=ax.get_lines()[-1].get_color(),
                alpha=0.7,
                linewidth=1.5,
                zorder=1,
                label = None
            )

            ax.text(
                x_vals[0],
                base_val * 1.0005,  # small vertical offset
                baseline_label,
                va="bottom",
                ha="left"
            )

    # ticks & formatting
    ax.set_xticks(x_vals)                                   # ticks exactly at PVs
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  
    ax.minorticks_on()                                      # enable minor ticks
    ax.tick_params(axis='x', which='major', length=6, width=1.2)
    ax.tick_params(axis='x', which='minor', length=4, color='gray')

    if not is_poster:
        ax.set_title(f"{plot_title} Peak Memory Consumption (Img-Size={img_dim}$\\times${img_dim})")

    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)

    if is_poster:
        ax.set_yscale("linear")

    ax.grid(
        True, 
        which="both", 
        linestyle="--", 
        alpha=0.4
    )
    ax.legend(
        frameon=False, 
        loc="upper left", # Can be adjusted based on data to "upper right", "upper left", etc.
        ncol=2 # Split the legend into 2 columns for easier viewing.
    )  # inside plot

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, 
            dpi=dpi, 
            format="png",
            bbox_inches="tight"
        )
        print(f"Saved figure to {save_path}")
    plt.show()

    return fig


def extract_peak_mem(file_path):
    df = pd.read_csv(file_path)

    peak_mem = df.peak_memory.item()

    return peak_mem

def main(args):

    mem_curves_save_dir = args.mem_curves_save_dir
    mem_dir = args.mem_dir
    exp_name = args.exp_name
    img_dim = args.img_dim
    model_name = args.model_name
    plot_name = args.plot_name
    xconv_varn = args.xconv_varn
    x_axis_name = args.x_axis_name
    y_axis_name = args.y_axis_name
    is_poster = args.is_poster
    pv_file_name = model_name + "_" + xconv_varn

    if not os.path.exists(mem_curves_save_dir):
        os.makedirs(mem_curves_save_dir)

    if args.batch_sizes is None:
        batch_sizes = [ 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        batch_sizes = args.batch_sizes

    probing_vectors = ['base', 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    data = []
    for batch_size in batch_sizes:
        row = []
        for probing_vector in probing_vectors:
            file_name = "{}_batch/{}_batch_size_{}_probing_vector_{}_peak_memory.csv".format(
                batch_size, pv_file_name, batch_size, probing_vector)
            file_path = os.path.join(mem_dir, file_name)

            if os.path.exists(file_path):
                constant = extract_peak_mem(file_path)
                row.append(constant)
            else:
                row.append(None)
                print(f"File not found: {file_path}")
        data.append(row)
        print(f"Processed batch size: {batch_size}")  


    df_heatmap = pd.DataFrame(data, index=batch_sizes, columns=probing_vectors)

    # Create a separate curve for each corresponding row ['batch_size']
    save_plot_name = '{}_img_dim_{}x{}_memory_curves.png'.format(exp_name,img_dim,img_dim)
    full_plot_path = '{}/{}'.format(mem_curves_save_dir,save_plot_name)

    dpi = 500 if is_poster else 300

    plot_memory_curves(
        df_heatmap=df_heatmap,
        probing_vectors=probing_vectors,
        batch_sizes=batch_sizes,
        img_dim=img_dim,
        save_path=full_plot_path,
        plot_title = plot_name,
        x_axis_name = x_axis_name,
        y_axis_name = y_axis_name,
        is_poster = is_poster,
        dpi = dpi
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
