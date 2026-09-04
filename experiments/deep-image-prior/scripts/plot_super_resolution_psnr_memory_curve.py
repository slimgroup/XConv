import argparse
import os
import matplotlib.pyplot as plt

"""
    Plot a PSNR vs Memory curve for different probing vectors.

    Usage:
        sh bash_scripts/bash_plot_psnr_memory_curve.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description= "Plot PSNR vs Memory curve for different probing vectors."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help = "Directory to save the final image.",
        required=True
    )
    return parser.parse_args()

def plot_psnr_vs_memory(
    pv_avg_psnr,
    pv_peak_mem,
    save_dir,
    base_psnr,
    base_mem
):

    """
        Plot the PSNR vs Memory curve for different probing vectors.
    """


    pv_vals = sorted(pv_peak_mem.keys())
    points = [(pv_peak_mem[pv], pv_avg_psnr[pv], pv) for pv in pv_vals]

    # Sort by Memory for plotting
    points.sort(key=lambda x: x[0])

    mem_vals = [x[0] for x in points]
    psnr_vals = [x[1] for x in points]
    pv_vals = [x[2] for x in points]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Scatter plot for PSNR vs Memory
    ax.scatter(
        mem_vals,
        psnr_vals,
        s = 60
    )

    label_offsets = {
        128: (-10, 6),
        256: (6, 6),
        512: (6, 2),
        1024: (-15, 10),
    }

    # Annotate the points with the probing vector values
    for mem, psnr, pv in points:
        dx, dy = label_offsets.get(pv, (5, 5))
        ax.annotate(
            f"r={pv}",
            (mem, psnr),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9,
            ha='left',
            va='bottom'
    )

    # Baseline point
    ax.scatter(
        [base_mem], [base_psnr],
        color='black',
        s=90,
        marker='o',
        label='Base (Exact Conv)',
        zorder=6
    )    

    # Horizontal line at the base PSNR
    ax.axhline(
        base_psnr, 
        color='black', 
        linestyle='--', 
        linewidth=1
    )

    # Vertical line at the base memory
    ax.axvline(
        base_mem, 
        color='black', 
        linestyle='--', 
        linewidth=1
    )

    # # Label the baseline crosshair
    # ax.annotate(
    #     "Exact Conv baseline",
    #     (base_mem, base_psnr),
    #     textcoords="offset points",
    #     xytext=(8, -18),   # tweak if it overlaps
    #     fontsize=9,
    #     fontweight='bold',
    #     ha='left',
    #     va='top'
    # )

    ax.set_xlabel("Peak Memory (MB)")
    ax.set_ylabel("PSNR")
    ax.set_title("PSNR–Peak Memory Tradeoff for DIP Super-Resolution")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()    

    plot_path = os.path.join(save_dir, "sr_psnr_memory_curve.png")
    print("Saving plot in {}".format(plot_path))

    plt.savefig(
        plot_path, 
        dpi=300, 
        format="png",
        bbox_inches="tight"
    )
    plt.close(fig)

    return

def main(args):

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_psnr = 28.58
    base_mem = 683.127

    pv_avg_psnr = {
        32: 25.27,
        64: 26.0,
        128: 26.66,
        256: 27.31,
        512: 27.70,
        1024: 28.14
    }

    pv_peak_mem = {
        32: 454.101,
        64: 453.861,
        128: 466.976,
        256: 557.364,
        512: 739.961,
        1024: 1109.154
    }

    plot_psnr_vs_memory(
        pv_avg_psnr = pv_avg_psnr, 
        pv_peak_mem = pv_peak_mem,
        save_dir = save_dir,
        base_psnr = base_psnr,
        base_mem = base_mem
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)