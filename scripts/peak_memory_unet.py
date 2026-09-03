"""RAD vs XConv on a U-Net: AGE vs image dimension, the paper's batch-budget protocol.

For each method (exact Conv, XConv with BitReLU at several probing-vector counts
r, RAD random-projection / sampling) and each input resolution, the batch size
is the MAXIMUM that fits a fixed memory budget (binary search) -- so the batch
shrinks as resolution grows, AGE rises accordingly, and XConv's lower memory
buys a larger batch (lower sampling noise) at the same resolution. AGE (Eq. 9)
is computed at that batch against the exact full-dataset gradient, with a
+/- sigma band over runs. Figures match the repo's AGE-figure theme, with the
max batch annotated on each point.

Config: rad_vs_xconv_unet_age.json

Train + visualize:  python scripts/peak_memory_unet.py
Visualize only:     python scripts/peak_memory_unet.py --phase visualization
"""

import os

# expandable_segments reduces the allocator fragmentation that caused the
# fp32 fwd+bwd to OOM near the budget (the chosen max batch had ~zero headroom).
# PYTORCH_CUDA_ALLOC_CONF is deprecated (a no-op) on torch 2.9; PYTORCH_ALLOC_CONF
# is the current name. Set BOTH, before importing torch, for old/new torch.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from projorg import checkpointsdir, plotsdir, setup_environment, upload_to_cloud

from pyxconv.radcompare import plotting, run_unet_age_vs_imgdim

CONFIG_FILE = "rad_vs_xconv_unet_age.json"


class RadVsXConvUNetAge:

    def __init__(self, args):
        self.args = args
        use_cuda = args.gpu_id >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.device = "cuda" if use_cuda else "cpu"

    def train(self):
        args = self.args
        print(f"[device] {self.device}", flush=True)
        ckpt_path = os.path.join(
            checkpointsdir(args.experiment), "checkpoint.pth"
        )
        # Incremental checkpoint + re-plot after EACH image dimension finishes
        # (run_unet_age_vs_imgdim calls this once a resolution is fully measured),
        # so figures appear as resolutions complete and a crash/OOM never loses an
        # already-finished resolution.
        plotting.apply_paper_style()

        def checkpoint(results):
            torch.save({"results": results, "args": args}, ckpt_path)
            self._render_figures(results)

        results = run_unet_age_vs_imgdim(
            img_dims=args.img_dims,
            r_list=args.r_list,
            keep_fracs=args.keep_fracs,
            budget_mib=args.mem_budget_gb * 1024.0,
            subset_size=args.subset_size,
            device=self.device,
            n_runs=args.n_runs,
            theta_seed=args.seed,
            base_seed=1000 * (args.seed + 1),
            base_channels=args.base_channels,
            depth=args.depth,
            precision=args.precision,
            checkpoint_cb=checkpoint,
        )
        # Final persist (also covers the empty-img_dims edge case).
        torch.save({"results": results, "args": args}, ckpt_path)

    def load_checkpoint(self):
        ckpt_path = os.path.join(
            checkpointsdir(self.args.experiment), "checkpoint.pth"
        )
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
        self.results = torch.load(ckpt_path, weights_only=False)["results"]

    def _render_figures(self, results):
        """Render all three figure types from ``results`` (shared by the
        incremental train-time re-plot and the standalone visualize phase).
        Tolerates partial results (a subset of resolutions measured so far): the
        peak curve only plots resolutions present in ``peak_curve``, and the
        AGE/max-batch plotters truncate the x-axis to the measured resolutions."""
        out = plotsdir(self.args.experiment)
        budget = self.args.mem_budget_gb
        prec = self.args.precision
        # Stage 1: the peak-memory curves that determine each method's max batch.
        for image_dim in self.args.img_dims:
            # Only plot resolutions that have measured peak curves yet.
            if not any(image_dim in r.get("peak_curve", {}) for r in results.values()):
                continue
            plotting.plot_peak_curve(
                results, image_dim, budget * 1024,
                os.path.join(out, f"unet_peak_curve_img{image_dim}_{prec}.pdf"),
                title=f"Peak memory vs batch ({prec}, Img-Size={image_dim}, "
                      f"{budget:g} GB budget)",
            )
        # Stage 2: AGE at those max batches, and the max-batch summary.
        plotting.plot_age_vs_imgdim_with_batches(
            results, os.path.join(out, f"unet_age_vs_imgdim_budget_{prec}.pdf"),
            title=f"U-Net AGE vs image dimension ({prec}, {budget:g} GB budget; "
                  f"batch annotated)",
        )
        plotting.plot_maxbatch_vs_imgdim(
            results, os.path.join(out, f"unet_maxbatch_vs_imgdim_{prec}.pdf"),
            title=f"Max batch within {budget:g} GB vs image dimension ({prec})",
        )

    def visualize(self):
        plotting.apply_paper_style()
        self._render_figures(self.results)


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[
            ("img_dims", int),
            ("r_list", int),
            ("keep_fracs", float),
        ],
    )

    experiment = RadVsXConvUNetAge(args)

    if args.phase == "train":
        experiment.train()

    experiment.load_checkpoint()
    experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
