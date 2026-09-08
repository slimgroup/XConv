"""Train an unconditional DDPM on the Parihaka seismic dataset.

By default reads ``configs/ddpm_128.json``. Set ``GENPARIHAKA_DDPM_CONFIG``
to point at a different config file in ``configs/`` (e.g. ``ddpm_256.json``).
Individual fields can be overridden from the command line by argparse, e.g.
``--batchsize 16 --max_epochs 50``.

Usage::

    python scripts/train_ddpm.py
    GENPARIHAKA_DDPM_CONFIG=ddpm_256.json python scripts/train_ddpm.py
    python scripts/train_ddpm.py --phase visualization
    python scripts/train_ddpm.py --upload 1
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from projorg import (
    checkpointsdir,
    plotsdir,
    setup_environment,
    upload_to_cloud,
)
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genparihaka import (
    SeismicDDPM,
    load_parihaka,
    plot_grid,
    plot_image,
    plot_losses,
)
from pyxconv.utils import convert_net

# CONFIG_FILE = os.environ.get("GENPARIHAKA_DDPM_CONFIG", "ddpm_128.json")
CONFIG_FILE = os.environ.get("GENPARIHAKA_DDPM_CONFIG", "xconv_configs/ddpm_128_pv512_lr_2e-4.json")

print("Using {}".format(CONFIG_FILE))

class DDPMExperiment:
    def __init__(self, args) -> None:
        self.args = args
        if torch.cuda.is_available() and args.gpu_id >= 0:
            self.device = torch.device(f"cuda:{args.gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.x_train, self.x_val, self.normalizer = load_parihaka(
            image_size=args.image_size,
            num_train=args.num_train,
            num_val=args.num_val,
            seed=args.seed,
        )
        print(f"Train: {tuple(self.x_train.shape)}, val: {tuple(self.x_val.shape)}")

        self.train_loader = DataLoader(
            TensorDataset(self.x_train),
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(self.x_val),
            batch_size=2 * args.batchsize,
            shuffle=False,
            drop_last=False,
        )

        self.model = SeismicDDPM(args).to(self.device)
        pv = args.pv

        print("Processing probing vector: {}".format(pv))
        base = (pv == 'base')

        if pv.isdigit():
            pv = int(pv)

        if not base:
            convert_net(
                self.model.model,
                ps = pv,
                xmode ='independent'
            )

        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"#Params \n{trainable_params / 1e6:.1f}M")
        print(self.model.model)

        self.train_obj: list = []
        self.val_obj: list = []
        self.start_epoch = 0

    def _ckpt_path(self, epoch) -> str:
        checkpoint_name = f"checkpoint_{epoch}.pth"
        return os.path.join(checkpointsdir(self.args.experiment), checkpoint_name)

    def _save(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.bundled_state_dict(),
                "normalizer": self.normalizer,
                "train_obj": self.train_obj,
                "val_obj": self.val_obj,
                "args": self.args,
            },
            self._ckpt_path(epoch),
        )

    def _render(self, epoch: int) -> None:
        plot_dir = plotsdir(self.args.experiment)
        gen = self.model.sample(self.args.num_samples, self.device)
        print("Storing individual generated images...")
        denorm_gen = self.normalizer.unnormalize(gen)
        gen_dir = os.path.join(plot_dir, f"generated_{epoch:04d}/")
        for i in range(self.args.num_samples):
            print(f"Saving individual generated image {i} of {self.args.num_samples}")
            single_gen_path = os.path.join(gen_dir, f"generated_{epoch:04d}_{i:04d}.png")
            print(f"Saving individual generated image at {single_gen_path}")
            plot_image(denorm_gen[i], single_gen_path)

        gen_path = os.path.join(plot_dir, f"generated_{epoch:04d}.png")        
        print("Saving grid of generated images at {}".format(gen_path))

        plot_grid(
            denorm_gen,
            gen_path,
            # title=f"Generated (epoch {epoch})"
        )
        real_path = os.path.join(plot_dir, f"real_{epoch:04d}.png")
        print("Saving grid of real images at {}".format(real_path))
        plot_grid(
            self.normalizer.unnormalize(self.x_train[: self.args.num_samples]),
            real_path,
            # title="Training samples"
        )
        print("Storing individual real images...")
        denorm_real = self.normalizer.unnormalize(self.x_train[: self.args.num_samples])
        real_dir = os.path.join(plot_dir, f"real_{epoch:04d}/")
        for i in range(self.args.num_samples):
            print(f"Saving individual real image {i} of {self.args.num_samples}")
            single_real_path = os.path.join(real_dir, f"real_{epoch:04d}_{i:04d}.png")
            print(f"Saving individual real image at {single_real_path}")
            plot_image(denorm_real[i], single_real_path)

        plot_losses(
            self.train_obj,
            self.val_obj,
            val_every=self.args.val_every,
            path=os.path.join(plot_dir, "log.png"),
        )

    def load_checkpoint(self) -> int:

        if hasattr(self.args, "ckpt_path"):
            path = self.args.ckpt_path
        else:
            path = self._ckpt_path()
        print(f"Loading checkpoint from {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_bundled_state_dict(ckpt["model"])
        self.normalizer = ckpt["normalizer"]
        self.train_obj = ckpt["train_obj"]
        self.val_obj = ckpt["val_obj"]
        self.start_epoch = ckpt["epoch"] + 1
        return ckpt["epoch"]

    def train(self) -> None:
        for epoch in tqdm(
            range(self.start_epoch, self.args.max_epochs),
            desc="epoch",
            dynamic_ncols=True,
        ):
            if epoch % self.args.val_every == 0:
                self.val_obj.append(self.model.val_epoch(self.val_loader, self.device))
            self.train_obj.extend(self.model.train_epoch(self.train_loader, self.device))
            last = epoch == self.args.max_epochs - 1
            if epoch % self.args.save_freq == 0 or last:
                self._save(epoch)
                self._render(epoch)

    def visualize(self) -> None:
        epoch = self.load_checkpoint()
        self._render(epoch)

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
            """
            Numpy implementation of the Frechet Distance.
            The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) is:
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
            """
            # Move to CPU and numpy for scipy.linalg operations
            mu1 = mu1.cpu().numpy()
            mu2 = mu2.cpu().numpy()
            sigma1 = sigma1.cpu().numpy()
            sigma2 = sigma2.cpu().numpy()

            diff = mu1 - mu2

            # Product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            # Numerical error might give slight imaginary component
            if not np.isfinite(covmean).all():
                print("FID calculation produced infinite values; returning NaN.")
                return np.nan

            if np.iscomplexobj(covmean):
                if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
                    print(f"m = {np.max(np.abs(covmean.imag))}")
                    print("Warning: FID calculation produced complex values.")
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
            return float(fid)


if __name__ == "__main__":
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "upload"],
        sequence_args_and_types=[("block_channels", int)],
    )

    experiment = DDPMExperiment(args)
    if args.phase == "train":
        experiment.train()
    else:
        experiment.visualize()

    if args.upload:
        upload_to_cloud(args)
