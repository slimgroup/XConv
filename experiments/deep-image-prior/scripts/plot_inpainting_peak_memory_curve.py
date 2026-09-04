import argparse
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

"""
    Plot the peak memory curve for the different probing vectors.

    Usage:
        sh bash_scripts/bash_plot_inpainting_peak_memory_curve.sh
"""

def parse_args():

    parser = argparse.ArgumentParser(description="Plot peak memory curve")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory where the final plots are saved."
    )
    parser.add_argument(
        "--mem_dir", 
        type=str, 
        required=True,
        help="Directory containing the csv files containing the inpainting model's peak memory."
    )
    parser.add_argument(
        "--save_plot_name",
        type=str,
        required=True,
        help="Name of the plot to save"
    )
    return parser.parse_args()

def plot_mem_curve(
    pv_mem_dict, 
    output_dir,
    save_plot_name
):

    """
        Plot the peak memory curve for the different probing vectors.
    """

    x_vals = sorted(int(pv) for pv in pv_mem_dict.keys() if pv != "base")
    mem_values = [pv_mem_dict[str(pv)] for pv in x_vals]    
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_val = pv_mem_dict['base']

    # Plot a horizontal line at the base value
    ax.axhline(
        y=base_val, 
        color='black', 
        linestyle='--', 
        label='Base',
        linewidth=2.2
    )

    ax.plot(
        x_vals,
        mem_values,
        '-o',
        lw=2,
        ms=6,
    )

    # Set the x-axis to a logarithmic scale with base 2
    ax.set_xscale('log', basex=2)


    # ticks & formatting
    ax.set_xticks(x_vals)                                   # ticks exactly at PVs
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  
    ax.minorticks_off()                                      
 
    # ax.set_title(f"{plot_title} (img_dim={img_dim}x{img_dim})")
    ax.set_title("Deep-Image-Prior: Peak Memory Consumption vs Probing Vectors (Inpainting)")
    ax.set_xlabel("Probing vectors $(r)$")
    ax.set_ylabel("Peak Memory (MB)")
    ax.grid(
        True, 
        which="both", 
        linestyle="--", 
        alpha=0.4
    )
    ax.legend(
        frameon=False, 
        loc="upper left", # Can be adjusted based on data to "upper right", "upper left", etc.
    )  # inside plot

    plt.tight_layout()

    save_path = os.path.join(output_dir, save_plot_name)
    print(f"Saving figure to {save_path}")

    plt.savefig(
        save_path, 
        dpi=300, 
        format="png",
        bbox_inches="tight"
    )
    plt.close(fig)
    return

def main(args):
    output_dir = args.output_dir
    mem_dir = args.mem_dir
    save_plot_name = args.save_plot_name

    assert os.path.exists(mem_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pv_mem_dict = {}

    for pv_path in os.listdir(mem_dir):
        pv = pv_path.split("or_")[-1]
        csv_path = os.path.join(mem_dir, pv_path, "peak_memory.csv")
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        mem_mib = df.iloc[0]['peak_memory']
        pv_mem_dict[pv] = mem_mib

    plot_mem_curve(pv_mem_dict, output_dir, save_plot_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)