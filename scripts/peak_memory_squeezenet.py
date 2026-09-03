"""Reproduce the paper's SqueezeNet peak-memory figure on this 16 GB GPU.

In-process, OOM-safe re-run of the repo's peak-memory experiment
(scripts/compute_peak_memory.py): torchvision squeezenet1_0, convert_net
(mode='all' -> XConv conv + BitReLU), measured with this repo's MemoryTracker
(torch_peak/2**20). Peak memory vs probing vectors, one line per batch size,
per image dimension, with exact convolution ('base') as a dashed reference --
the paper's Fig-8 curves. CUDA-OOM configs are recorded as gaps (the paper's
'infeasible' regime) rather than crashing, so the whole grid runs within 16 GB.

The ``precision`` config field selects fp32 (the original) or fp16 (model+inputs
.half(), matching the AGE scripts' ``--bf16_precision`` path) -- the fp16 Fig-8
panels. fp16 lands in its OWN experiment dir (separate ``experiment_name`` in
configs/squeezenet_peak_memory_fp16.json), so fp32 results are never touched.

Config: squeezenet_peak_memory.json (fp32) | squeezenet_peak_memory_fp16.json (fp16)

Train + visualize:  python scripts/peak_memory_squeezenet.py
fp16 panels:        python scripts/peak_memory_squeezenet.py \
                        --config squeezenet_peak_memory_fp16.json
Visualize only:     python scripts/peak_memory_squeezenet.py --phase visualization
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# PYTORCH_CUDA_ALLOC_CONF is deprecated on torch>=2.9; set the new name too so
# expandable_segments (anti-fragmentation) actually takes effect. Methodology-neutral.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from projorg import checkpointsdir, plotsdir, setup_environment, upload_to_cloud

from pyxconv.radcompare import plotting
from pyxconv.radcompare.sweeps import run_squeezenet_peak_memory_sweep

# Default fp32 config. Override with `--config <file.json>` (consumed before
# setup_environment, since projorg builds its CLI parser from the JSON keys and
# would otherwise reject an unknown --config flag). Use
# squeezenet_peak_memory_fp16.json for the half-precision Fig-8 panels.
CONFIG_FILE = "squeezenet_peak_memory.json"
if "--config" in sys.argv:
    _i = sys.argv.index("--config")
    CONFIG_FILE = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]  # strip so projorg's JSON-derived parser never sees it


class SqueezenetPeakMemory:

    def __init__(self, args):
        self.args = args
        use_cuda = args.gpu_id >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
        self.device = "cuda" if use_cuda else "cpu"
        # 'base' stays a string; numeric probing vectors are kept as strings too
        # (the sweep does int(pv) for non-'base').
        self.probing_vectors = [p.strip() for p in args.probing_vectors.split(",")]
        # 'fp32' (default) or 'fp16'. fp16 -> model+inputs .half() in the sweep.
        # Absent from older configs -> default fp32.
        self.precision = getattr(args, "precision", "fp32")

    def train(self):
        args = self.args
        print(f"[device] {self.device}  precision={self.precision}", flush=True)
        ckpt_path = os.path.join(
            checkpointsdir(args.experiment), "checkpoint.pth"
        )
        # Incremental checkpoint + re-plot after EVERY measured config (each
        # (pv, batch) within each image dim), so figures appear early and a
        # crash/OOM never loses already-measured configs. The sweep is OOM-safe
        # (NaN for configs exceeding the GPU), so this just runs to completion.
        plotting.apply_paper_style()
        out = plotsdir(args.experiment)
        done = []  # rows for image dims already fully finished

        def checkpoint(current):
            """Persist + re-plot all rows finished so far (finished dims + the
            in-progress dim's rows-so-far)."""
            records = done + current
            torch.save({"records": records, "args": args}, ckpt_path)
            for image_dim in sorted({r["img_dim"] for r in records}):
                plotting.plot_peak_vs_probing(
                    records, image_dim,
                    os.path.join(out, f"squeezenet_peak_memory_img{image_dim}.pdf"),
                )

        for image_dim in args.img_dims:
            # One image dim at a time; checkpoint_cb fires after each (pv, batch).
            dim_records = run_squeezenet_peak_memory_sweep(
                img_dims=[image_dim],
                batch_sizes=args.batch_sizes,
                probing_vectors=self.probing_vectors,
                device=self.device,
                checkpoint_cb=checkpoint,
                precision=self.precision,
            )
            done += dim_records
            checkpoint([])
            print(f"[checkpoint] img_dim={image_dim} done; "
                  f"checkpoint + figures updated at {out}", flush=True)

    def load_checkpoint(self):
        ckpt_path = os.path.join(
            checkpointsdir(self.args.experiment), "checkpoint.pth"
        )
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
        self.records = torch.load(ckpt_path, weights_only=False)["records"]

    def visualize(self):
        out = plotsdir(self.args.experiment)
        plotting.apply_paper_style()
        for image_dim in self.args.img_dims:
            plotting.plot_peak_vs_probing(
                self.records, image_dim,
                os.path.join(out, f"squeezenet_peak_memory_img{image_dim}.pdf"),
            )


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[
            ("img_dims", int),
            ("batch_sizes", int),
        ],
    )

    experiment = SqueezenetPeakMemory(args)

    if args.phase == "train":
        experiment.train()

    experiment.load_checkpoint()
    experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
