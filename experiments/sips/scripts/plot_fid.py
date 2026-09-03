import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

"""

Plot FID values for different probing vectors.

Usage:
    sh bash_scripts/plot_fid.sh
"""

def extract_int(
    input_str
):
    """
    Extract the probing vector from the x-axis name.
    """
    match = re.search(r"\d+", input_str)

    if match:
        pv_str = match.group()

        pv = int(pv_str)

        return pv    
    
    return None

def plot_fid_values(
    fid_dict_run1: dict,
    fid_dict_run2: dict,
    fid_dict_run3: dict,
    save_path: str,
):
    """
    Plot FID values for different probing vectors and different runs in a single plot.
    Args:
        fid_dict_run1: Dictionary containing FID values for different probing vectors for run 1.
        fid_dict_run2: Dictionary containing FID values for different probing vectors for run 2.
        fid_dict_run3: Dictionary containing FID values for different probing vectors for run 3.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # ---- Baseline FID ----
    # Plot 3 baseline lines for the 3 runs
    runs = [
        (fid_dict_run1, "Run 1", "blue"),
        (fid_dict_run2, "Run 2", "red"),
        (fid_dict_run3, "Run 3", "green")
    ]

    for fid_dict, label, color in runs:
        base_fid = fid_dict["Conv"]
        ax.axhline(
            y=base_fid,
            linestyle='--',
            color=color,
            label=f'Baseline Conv {label}',
            alpha=0.8,
            lw=1.5
        )

    # ---- Probing vector curves ----
    # Get x-axis names from any run (assuming all runs have same keys)
    x_axis_names = [k for k in fid_dict_run1.keys() if k != "Conv"]

    # Extract the PVs from x_axis_names.
    pv_list = [extract_int(name) for name in x_axis_names]

    for fid_dict, label, color in runs:
        y_values = [fid_dict[x] for x in x_axis_names]
        ax.plot(
            pv_list,
            y_values,
            marker = 'o',
            color = color,
            label = label,
            lw=2,
            ms=6,
        )

    # ticks & formatting

    # ticks exactly at PVs
    ax.set_xticks(pv_list)                                   
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  

    # enable minor ticks
    ax.minorticks_on()                                      
    ax.tick_params(
        axis='x', 
        which='major', 
        length=6, 
        width=1.2
    )
    ax.tick_params(
        axis='x', 
        which='minor', 
        length=4, 
        color='gray'
    )

    # ---- Labels, title, grid ----
    # ax.set_title("FID vs Probing Vector", pad=10)
    ax.set_xlabel("Probing Vectors")
    ax.set_ylabel("FID")

    ax.grid(
        True, 
        which="both", 
        linestyle="--", 
        alpha=0.4
    )

    ax.legend(
        frameon=True, 
        loc='best', 
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(
        save_path, 
        bbox_inches="tight", 
        format="png",
        dpi=300
    )
    plt.close(fig)
    print(f"Saved plot to {save_path}")
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Plot FID values for different probing vectors and different runs in a single plot.')
    parser.add_argument(
        "--plot_dir", 
        type=str, 
        required=True, 
        help='Directory to save the plot.')
    parser.add_argument(
        "--plot_name", 
        type=str, 
        required=True, 
        help='Name of the plot.')
    return parser.parse_args()

def main(args):
    plot_dir = args.plot_dir
    plot_name = args.plot_name

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fid_dict_run1 = {
        "Conv": 0.045,
        "PV-32": 0.154,
        "PV-64": 0.115,
        "PV-128": 0.149,
        "PV-256": 0.050,
    }
    fid_dict_run2 = {
        "Conv": 0.032,
        "PV-32": 0.111,
        "PV-64": 0.052,
        "PV-128": 0.074,
        "PV-256": 0.053,
    }
    fid_dict_run3 = {
        "Conv": 0.063,
        "PV-32": 0.117,
        "PV-64": 0.066,
        "PV-128": 0.075,
        "PV-256": 0.072,
    }

    save_path = os.path.join(plot_dir, plot_name)

    # prepare plot
    plot_fid_values(
        fid_dict_run1=fid_dict_run1,
        fid_dict_run2=fid_dict_run2,
        fid_dict_run3=fid_dict_run3,
        save_path=save_path
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)