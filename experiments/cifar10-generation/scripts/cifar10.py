# pylint: disable=E1102, invalid-name
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import argparse
import pickle
import re
import torch
import torchvision
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

# Diffusers / Scheduler imports
from diffusers import DDPMScheduler, DDIMScheduler

# Local project imports
from projorg import (
    checkpointsdir, 
    datadir, 
    setup_environment, 
    upload_to_cloud
)
from sips.utils import plot_loss, plotsdir
from sips.models import UNet2DModel 

# Torch Data / Optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from only_comp_avg_grad_err import get_conv_param_names, compute_avg_grad_err
from pyxconv.utils import convert_net

# Change the config file here.
CONFIG_FILE = os.environ.get("SIPS_CONFIG_FILE", "cifar10_example.json")
# CONFIG_FILE = os.environ.get("SIPS_CONFIG_FILE", "xconv_configs/cifar10_pv2_example.json")
print(f"Using config file: {CONFIG_FILE}")

_num = re.compile(r'(\d+)')

def natural_key(fname: str) -> int:
    m = _num.search(fname)
    if not m:
        raise ValueError(f"No integer found in {fname}")
    return int(m.group(1))

def set_seed(seed: int) -> None:
    """Set the seed for the random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class StoredDataset(torch.utils.data.Dataset):
    """
        A dataset that returns the stored Images, z's and t's.
    """
    def __init__(
          self, 
          img_folder, 
          z_folder,
          t_folder,
          subset_size,
          device
    ):
        self.img_folder = img_folder
        self.z_folder = z_folder
        self.t_folder = t_folder
        self.device = device
        self.subset_size = subset_size
        self.num_files = subset_size

        assert self.num_files == self.subset_size

        # Sort for deterministic order
        # List only the relevant files, then sort numerically by the embedded integer
        img_files = [f for f in os.listdir(img_folder) if f.endswith('.pt') and f.startswith('img_')]
        img_files.sort(key=natural_key)  # <- numeric sort
        self.img_files = img_files[:subset_size]

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Extract image index from filename
        file_name = self.img_files[idx]
        base_name = os.path.splitext(file_name)[0]  # "image_716973"
        file_idx = base_name.split("_")[-1]         # "716973"

        # (C, H, W) eg. (3, 1024, 1024)
        img_path = os.path.join(self.img_folder, "img_{}.pt".format(file_idx))
        img = torch.load(img_path)

        # t: (1) eg. (1)

        t_path = os.path.join(self.t_folder, f"t_{file_idx}.pt")
        t = torch.load(t_path)

        z_path = os.path.join(self.z_folder, f"z_{file_idx}.pt")
        z = torch.load(z_path)

        return img.squeeze(0), z.squeeze(0), t.squeeze(0)


class CIFAR10Example:
    """A class demonstrating the CIFAR10 example.

    Attributes:
        device (torch.device): The device (cpu/cuda) used for computation.
        score_model: The score neural network.
        optimizer: The optimizer used for training.
        train_obj: Placeholder for training objective values.
        val_obj: Placeholder for validation objective values.
    """

    def warmup_schedule(self, step: int) -> float:
        if step < self.args.warmup_steps:
            return step / self.args.warmup_steps
        return 1.0
    
    
    def __init__(self, args: argparse.ArgumentParser) -> None:
            """
            Initialize the CIFAR10Example object with subset datasets.

            Args:
                args (argparse.ArgumentParser): The command line arguments.
            """
           
            self.args = args
            if torch.cuda.is_available() and args.gpu_id > -1:
                self.device = torch.device(f"cuda:{args.gpu_id}")
            else:
                self.device = torch.device("cpu")

            if args.holdout < 0.0 or args.holdout > 1.0:
                raise ValueError("Holdout fraction must be in [0.0, 1.0]")

            self.train_penalty_obj = {"tk": [], "holdout": []}
            self.val_penalty_obj = {"tk": [], "holdout": []}
            self.ratio_mem_metric = []

            self.plot_dir = plotsdir(args.experiment)
            os.makedirs(self.plot_dir, exist_ok=True)

            self.x_train = torchvision.datasets.CIFAR10(
                datadir("datasets"),
                train=True,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))
                ])
            )

            # Load CIFAR10 validation dataset
            x_val = torchvision.datasets.CIFAR10(
                datadir("datasets"),
                train=False,
                download=False,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))
                ])
            )

            # DataLoaders
            self.train_loader = DataLoader(
                self.x_train,
                batch_size=args.batchsize,
                shuffle=True,
                drop_last=False,
                num_workers=1,
            )
            self.val_loader = DataLoader(
                x_val,
                batch_size=args.batchsize,
                shuffle=False,
                drop_last=False,
                num_workers=1,
            )

            self.score_model = UNet2DModel(
                in_channels=3,
                out_channels=3,
                sample_size=(32, 32),
                time_embedding_type=args.time_emb,
                attention_head_dim=None,
                block_out_channels=(128, 256, 256, 256),
                layers_per_block=2,
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
                dropout=0.1,
                norm_num_groups=32,
            ).to(self.device)

            if hasattr(args, "pv"):
                probing_vector = args.pv
            else:
                probing_vector = "base"

            # for probing_vector in probing_vectors:
            print("Processing probing_vector: {}".format(probing_vector))
            base = (probing_vector == 'base')

            if probing_vector.isdigit():
                probing_vector = int(probing_vector)

            self.probing_vector = probing_vector

            if not base: 
                convert_net(
                    self.score_model, 
                    ps = probing_vector,
                    xmode ='independent'
                )

            trainable_params = sum(p.numel() for p in self.score_model.parameters() if p.requires_grad)
            print(f"#Params \n{trainable_params / 1e6:.1f}M")

            print(self.score_model)

            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.nt,
                beta_schedule=args.beta_schedule,
            )

            self.optimizer = torch.optim.Adam(
                self.score_model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            self.lr_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=self.warmup_schedule
            )

            # Placeholders and EMA
            self.train_obj = []
            self.val_obj = []
            self.ema_model = deepcopy(self.score_model)
            self.ema_decay = 0.9999

    def load_checkpoint(
            self, 
            args: argparse.Namespace, 
            epoch: int | None = None
        ) -> int:
        ckpt_dir = checkpointsdir(args.experiment)
        os.makedirs(ckpt_dir, exist_ok=True)

        if hasattr(args, "ckpt_path"):
            file_to_load = args.ckpt_path

            if not os.path.exists(file_to_load):
                print(f"Checkpoint not found: {file_to_load}")
                return 0
        else:
            if epoch is not None:
                file_to_load = os.path.join(
                    ckpt_dir, f"checkpoint_{epoch}.pth"
                )

                if not os.path.exists(file_to_load):
                    print(f"Checkpoint not found: {file_to_load}")
                    return 0

            else:
                ckpt_files = [
                    f for f in os.listdir(ckpt_dir)
                    if f.startswith("checkpoint_") and f.endswith(".pth")
                ]

                if not ckpt_files:
                    print("No checkpoint found, starting from scratch")
                    return 0

                latest_ckpt = max(
                    ckpt_files,
                    key=lambda f: int(f.split("_")[-1].split(".")[0])
                )

                file_to_load = os.path.join(ckpt_dir, latest_ckpt)

        print(f"Loading checkpoint: {file_to_load}")

        checkpoint = torch.load(
            file_to_load,
            map_location=self.device,
            weights_only=False
        )

        self.score_model.load_state_dict(checkpoint["model_state_dict"])

        if "optim_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optim_state_dict"])

        self.train_obj = checkpoint.get("train_obj", [])
        self.val_obj = checkpoint.get("val_obj", [])
        self.train_penalty_obj = checkpoint.get(
            "train_penalty_obj", {"tk": [], "holdout": []}
        )
        self.val_penalty_obj = checkpoint.get(
            "val_penalty_obj", {"tk": [], "holdout": []}
        )
        self.ratio_mem_metric = checkpoint.get("ratio_mem_metric", [])

        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resumed from epoch {start_epoch}")

        return start_epoch


    def train(self, args: argparse.ArgumentParser) -> None:
        """Trains the hypernetwork.

        Args:
            args (argparse.ArgumentParser): The command line arguments.
        """
        start_epoch = self.load_checkpoint(args)

        for epoch in tqdm(
            range(start_epoch, args.max_epochs),
            unit="epoch",
            colour="#B5F2A9",
            dynamic_ncols=True,
            desc="Training progress",
        ):
            # Validation phase
            self.score_model.eval()

            if args.holdout > 0.0:
                holdout_noise_flat = torch.from_numpy(
                    self.cluster_aligner.sample_noise(
                        self.x_holdout_labels, rigidness=1.0
                    )
                ).to(self.device)
                holdout_noise = holdout_noise_flat.view(self.holdout_tensor.shape)

            with torch.no_grad():
                self.val_obj.append(0.0)

                if args.tk_reg > 0.0:
                    self.val_penalty_obj["tk"].append(0.0)
                if args.holdout > 0.0:
                    self.val_penalty_obj["holdout"].append(0.0)

                for x_val, _ in self.val_loader:
                    x_val = x_val.to(self.device)

                    noise = torch.randn(
                        x_val.shape,
                        device=self.device,
                    )

                    timesteps = torch.randint(
                        0,
                        len(self.noise_scheduler),
                        (x_val.shape[0],),
                        device=self.device,
                    ).long()

                    x_val_t = self.noise_scheduler.add_noise(
                        x_val,
                        noise,
                        timesteps,
                    )

                    noise_pred = self.score_model(
                        x_val_t,
                        timesteps,
                        return_dict=False,
                    )[0]

                    obj = torch.norm(noise_pred - noise) ** 2

                    self.val_obj[-1] += obj / args.batchsize

                if args.holdout > 0.0:
                    timesteps_ho = torch.randint(
                        0,
                        len(self.noise_scheduler),
                        (len(self.holdout_tensor),),
                        device=self.device,
                    ).long()

                    xt_ho = self.noise_scheduler.add_noise(
                        self.holdout_tensor,
                        holdout_noise,
                        timesteps_ho,
                    )
                    noise_pred_ho = self.score_model(
                        xt_ho, timesteps_ho, return_dict=False
                    )[0]
                    holdout_penalty = (
                        torch.norm(noise_pred_ho - holdout_noise) ** 2
                        / len(self.holdout_tensor)
                    )
                    self.val_obj[-1] += holdout_penalty
                    self.val_penalty_obj["holdout"][-1] = (
                        holdout_penalty.item()
                    )

                # Average and store validation objective.
                self.val_obj[-1] = self.val_obj[-1].item() / len(
                    self.val_loader
                )

                if args.holdout > 0.0:
                    self.val_penalty_obj["holdout"][-1] = (
                        self.val_penalty_obj["holdout"][-1]
                        / len(self.val_loader)
                    )

            # Training phase.
            self.score_model.train()

            with tqdm(
                self.train_loader,
                unit="iteration",
                colour="#B5F2A9",
                dynamic_ncols=True,
                desc="Epoch progress",
            ) as pb:
                for x_train, _ in pb:
                    x_train = x_train.to(self.device)

                    noise = torch.randn(
                        x_train.shape,
                        device=self.device,
                    )

                    # Randomly sample timesteps.
                    timesteps = torch.randint(
                        0,
                        len(self.noise_scheduler),
                        (x_train.shape[0],),
                        device=self.device,
                    ).long()

                    # Add noise to the data according to the noise schedule.
                    x_t = self.noise_scheduler.add_noise(
                        x_train,
                        noise,
                        timesteps,
                    )

                    # Predict the score at this noise level.
                    noise_pred = self.score_model(
                        x_t,
                        timesteps,
                        return_dict=False,
                    )[0]

                    # Calculate objective function.
                    obj = torch.norm(noise_pred - noise) ** 2
                    obj = obj / args.batchsize

                    # Backpropagation.
                    obj.backward()
                    torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        for ema_param, param in zip(self.ema_model.parameters(), self.score_model.parameters()):
                            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

                    # Update learning rate.
                    self.lr_scheduler.step()

                    # Store training objective.
                    self.train_obj.append(obj.item())

                    # Update progress bar.
                    pb.set_postfix(
                        {
                            "train obj": f"{self.train_obj[-1]:.2f}",
                            "val obj": f"{self.val_obj[-1]:.2f}",
                        }
                    )

            # Test the model.
            if epoch % args.save_freq == 0 or epoch == args.max_epochs -1:
                self.test(args, epoch=epoch)

            # Save model checkpoints.
            if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                torch.save(
                    {
                        "model_state_dict": self.score_model.state_dict(),
                        "ema_state_dict": self.ema_model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                        "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "train_obj": self.train_obj,
                        "val_obj": self.val_obj,
                        "train_penalty_obj": self.train_penalty_obj,
                        "val_penalty_obj": self.val_penalty_obj,
                    },
                    os.path.join(
                        checkpointsdir(args.experiment),
                        "checkpoint_" + str(epoch) + ".pth",
                    ),
                )



    @torch.no_grad()
    def ddim_sample_images_for_mem_metric(self, batch_size, sample_steps=None, eta=0.0):
        """
        DDIM sampling from a trained DDPM model.

        Args:
            batch_size: number of images
            sample_steps: number of DDIM steps (default: full schedule)
            eta: 0.0 = deterministic DDIM, >0 adds stochasticity
        """

        # ----------------------------
        # Build DDIM scheduler from DDPM config
        # ----------------------------
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        ddim_scheduler.set_timesteps(sample_steps if sample_steps is not None else self.noise_scheduler.config.num_train_timesteps)

        # ddim_scheduler = ddim_scheduler.to(self.device)

        # ----------------------------
        # Start from pure noise
        # ----------------------------
        x = torch.randn(batch_size, 3, 32, 32, device=self.device)

        # ----------------------------
        # DDIM reverse process
        # ----------------------------
        for t in ddim_scheduler.timesteps:
            t_batch = torch.full(
                (batch_size,),
                t,
                device=self.device,
                dtype=torch.long
            )

            # Predict noise (same model as DDPM training)
            noise_pred = self.score_model(
                x,
                t_batch,
                return_dict=False
            )[0]

            # DDIM step
            x = ddim_scheduler.step(
                noise_pred,
                t,
                x,
                eta=eta
            ).prev_sample

        return x
    
    def sample_images_for_mem_metric(self, batch_size):
        """
        batch_size: number of images to sample
        returns: sampled images tensor in [-1,1], shape (B,3,32,32)
        """
        timesteps = list(range(len(self.noise_scheduler)))[::-1]

        x = torch.randn(
            batch_size, 3, 32, 32, device=self.device
        )

        for t in timesteps:
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )
            residual = self.score_model(
                x, t_batch, return_dict=False
            )[0]
            x = self.noise_scheduler.step(
                residual, t, x
            ).prev_sample

        return x
    
    @torch.no_grad()
    def test(
        self, 
        args: argparse.Namespace, 
        epoch: int = -1
    ) -> None:
        """
        Full test routine:
        1) Plot training / validation losses
        2) Save a grid of generated samples
        """

        if epoch == -1:
            self.load_checkpoint(args)
            epoch = args.testing_epoch

        self.score_model.eval()

        plot_root = plotsdir("tmlr_" + args.experiment)
        os.makedirs(plot_root, exist_ok=True)

        # -------------------------------------------------
        # Generate samples in a five-row grid.
        # -------------------------------------------------
        gen = self.ddim_sample_images_for_mem_metric(
            batch_size=args.num_gen_imgs, 
            sample_steps=100, 
            eta=0.0
        )
        gen = (gen * 0.5 + 0.5).clamp(0, 1)

        # Lanczos enlargement for smoother 256x256 display tiles.
        gen = torch.stack([
            torchvision.transforms.functional.to_tensor(
                torchvision.transforms.functional.to_pil_image(image.cpu()).resize(
                    (256, 256),
                    resample=Image.Resampling.LANCZOS,
                )
            )
            for image in gen
        ])

        gen_path = os.path.join(
            plot_root, f"{epoch:04d}_generated.png"
        )

        images_per_row = int(np.ceil(args.num_gen_imgs / 5))
        torchvision.utils.save_image(
            gen,
            gen_path,
            nrow=images_per_row,
            padding=2,
        )

        print(f"Saved generated samples to: {gen_path}")

        # -------------------------------------------------
        # 6. Plot losses
        # -------------------------------------------------
        plot_loss(
            args,
            self.train_obj,
            self.val_obj,
            epoch
        )

    def comp_full_gradient_dict(
        self,
        score_model, 
        data_loader, 
        noise_scheduler,
        loss_reduction,
        device
    ):
        """
        Computes the full dataset gradient for the UNet[SIPs] by accumulating gradients over all mini-batches.

        This function performs a forward and backward pass for each mini-batch,
        scales each batch loss appropriately so the accumulated gradients reflect
        the true average gradient over the entire dataset.

        Args:
            score_model (torch.nn.Module): The model whose gradients are to be computed.
            data_loader (DataLoader): DataLoader for the full dataset.
            loss_reduction (str): The reduction type for the loss.
            noise_scheduler (DDPMScheduler): The noise scheduler used in the diffusion process.
            device (torch.device or str): Device to run computation on (e.g., 'cuda' or 'cpu').

        Returns:
            dict: A dictionary mapping parameter names to their accumulated gradient tensors.
                Gradients are detached from the graph and cloned.
        """

        score_model.train()
        score_model.zero_grad() # Erase gradients before performing this computation.
        grad_dict = {}

        total_samples = len(data_loader.dataset)
        
        for images, zs, ts in tqdm(data_loader, desc="Computing Full Gradient"):
            
                _ = images.size(0)

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                images = images.to(device)
                
                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                noise = zs.to(device)

                # B eg. 5
                timesteps = ts.long().to(device)

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                noisy_images = noise_scheduler.add_noise(
                    images,
                    noise,
                    timesteps,
                )

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                noise_pred = score_model(
                    noisy_images,
                    timesteps,
                    return_dict=False,
                )[0]

                # loss = torch.norm(noise_pred - noise) ** 2
                # loss = loss.mean()

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                diff = noise_pred - noise

                # per-sample loss: average over CHW for each item
                # (B,) eg. (5,)
                per_sample = diff.pow(2).flatten(1).mean(dim=1)   # shape (B,)

                # if loss_reduction == "sum":
                #     loss /= total_samples  # Scale loss to reflect full dataset
                # elif loss_reduction == "mean":
                #     loss /= batch_size
                # else:
                #     raise ValueError(f"Invalid loss reduction: {loss_reduction}")
                if loss_reduction == "mean":
                    # unbiased estimate of dataset mean gradient
                    loss = per_sample.mean()
                elif loss_reduction == "sum":
                    # unbiased estimate of dataset *sum* gradient; scale to dataset average
                    loss = per_sample.sum() / total_samples
                else:
                    raise ValueError("Invalid loss_reduction")


                loss.backward()  # Accumulates gradients

        # Extract and clone gradients
        for name, param in score_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_dict[name] = param.grad.clone().cpu().detach()

        return grad_dict
    
    def comp_mini_batch_and_avg_grad_err(
        self,
        score_model,
        data_loader,
        conv_param_names,
        full_grad_param_dict,   
        noise_scheduler,
        loss_reduction,
        device
    ):  
        """
        Computes the mini-batch gradients for the UNet[SIPs] and uses it to compute the average gradient error.

        This function performs a forward and backward pass for each mini-batch,
        accumulates the gradients, and then computes the average gradient error
        with respect to the provided full dataset gradients.

        Args:
            score_model: The neural network model (UNet[SIPs]) for which gradients are computed.
            data_loader: DataLoader providing mini-batches of data.
            conv_param_names: Set of parameter names corresponding to convolutional layers.
            full_grad_param_dict: Dictionary containing full dataset gradients for comparison.
            loss_reduction: The reduction type for the loss.
            device: The device (CPU or GPU) on which computations are performed.

        
        Returns:    
            avg_grad_err: The average L2 gradient error across all convolutional parameters.
        """
        score_model.train() 
        score_model.zero_grad()  
        batch_errors = []

        for i, (images, zs, ts) in enumerate(tqdm(data_loader, desc="Computing Mini-Batch Gradients")):
                
                score_model.zero_grad() # Zero the gradients before each batch

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                images = images.to(device)

                batch_size = images.size(0)
                
                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                noise = zs.to(device)

                # B eg. 5
                timesteps = ts.long().to(device)

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                noisy_images = noise_scheduler.add_noise(
                    images,
                    noise,
                    timesteps,
                )

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                noise_pred = score_model(
                    noisy_images,
                    timesteps,
                    return_dict=False,
                )[0]

                # loss = torch.norm(noise_pred - noise) ** 2
                # loss = loss.mean()

                # if loss_reduction == "sum":
                #     loss /= batch_size  # Scale loss to reflect mini-batch.
                # elif loss_reduction == "mean":
                #     pass # Loss is already scaled by batch size.
                # else:
                #     raise ValueError(f"Invalid loss reduction: {loss_reduction}")

                # (B, C, H, W) eg. (5, 1, 1024, 1024)
                diff = noise_pred - noise

                # per-sample loss: average over CHW for each item
                # (B,) eg. (5,)
                per_sample = diff.pow(2).flatten(1).mean(dim=1)   # shape (B,)

                # if loss_reduction == "sum":
                #     loss /= total_samples  # Scale loss to reflect full dataset
                # elif loss_reduction == "mean":
                #     loss /= batch_size
                # else:
                #     raise ValueError(f"Invalid loss reduction: {loss_reduction}")
                if loss_reduction == "mean":
                    # unbiased estimate of dataset mean gradient
                    pass
                elif loss_reduction == "sum":
                    # unbiased estimate of dataset *sum* gradient; scale to dataset average
                    loss = per_sample.sum() / batch_size
                else:
                    raise ValueError("Invalid loss_reduction")


                loss.backward()
                
                batch_grads = {
                    name: param.grad.clone().cpu().detach()
                    for name, param in score_model.named_parameters()
                    if param.requires_grad and param.grad is not None
                }

                # Get the gradient error for this batch.
                batch_grad_err = compute_avg_grad_err(
                    mini_batch_grads = {i : batch_grads},
                    full_model_grads = full_grad_param_dict,
                    model_layer = None,
                    conv_param_names= conv_param_names,
                )
                print("For batch {}, the gradient error is: {}".format(i, batch_grad_err))
                batch_errors.append(batch_grad_err)
        
        return np.mean(batch_errors)

    
    def compute_mini_batch_gradients(self, args: argparse.ArgumentParser) -> None:
        """
        Compute the mini-batch gradients and the average gradient error.
        
        Args:
            args (argparse.ArgumentParser): The command line arguments.
        """

        img_folder = args.img_folder
        z_folder = args.z_folder
        t_folder = args.t_folder
        subset_size = args.subset_size
        batch_size = args.batchsize
        full_grads_path = args.full_grads_path
        grad_dict_dir = args.grad_dict_dir

        if not os.path.exists(grad_dict_dir):
            os.makedirs(grad_dict_dir)

        print("Reading full gradients from {}".format(full_grads_path))
        with open(full_grads_path, "rb") as f:
            full_grad_param_dict = pickle.load(f)

        # Create a stored dataset that returns the Images, t's and z's.
        stored_dataset = StoredDataset(
            img_folder = img_folder,
            z_folder = z_folder,
            t_folder = t_folder,
            subset_size = subset_size,
            device = self.device
        )
    
        conv_param_names = get_conv_param_names(self.score_model)

        mini_batch_grad_loader = torch.utils.data.DataLoader(
            stored_dataset, 
            batch_size=batch_size
        )

        loss_reduction = "sum"

        # Compute mini-batch gradients and use it to compute the average gradient error.
        avg_err = self.comp_mini_batch_and_avg_grad_err(
            score_model = self.score_model,
            data_loader = mini_batch_grad_loader,
            conv_param_names=conv_param_names,
            full_grad_param_dict=full_grad_param_dict,
            noise_scheduler  = self.noise_scheduler,
            loss_reduction = loss_reduction,
            device = self.device
        )

        print(f"Avg L2 gradient error: {avg_err}")

        avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
        print("Saving average gradient error at: {}".format(avg_grad_err_file_path))
        with open(avg_grad_err_file_path, "wb") as f:
            pickle.dump(avg_err, f, pickle.HIGHEST_PROTOCOL)



        return
    
    def compute_full_gradients(self, args: argparse.ArgumentParser) -> None:
        """
        Compute the full gradient needed for the average gradient error computation.
        
        Args:
            args (argparse.ArgumentParser): The command line arguments.
        """

        img_folder = args.img_folder
        z_folder = args.z_folder
        t_folder = args.t_folder
        subset_size = args.subset_size
        batch_size = args.batchsize
        base_grad_dict_dir = args.base_grad_dict_dir

        if not os.path.exists(base_grad_dict_dir):
            os.makedirs(base_grad_dict_dir)

        # Create a stored dataset that returns the Images, t's and z's.
        stored_dataset = StoredDataset(
            img_folder = img_folder,
            z_folder = z_folder,
            t_folder = t_folder,
            subset_size = subset_size,
            device = self.device
        )
        
        full_grad_loader = torch.utils.data.DataLoader(
            stored_dataset, 
            batch_size=batch_size
        )

        loss_reduction = "sum"

        #  Compute the full gradients and store the corresponding model weights in a dictionary.      
        full_grad_param_dict = self.comp_full_gradient_dict(
            score_model = self.score_model,
            data_loader = full_grad_loader,
            noise_scheduler=self.noise_scheduler,
            loss_reduction = loss_reduction,
            device = self.device
        )

        final_path = os.path.join(base_grad_dict_dir, "full_grad_param_dict.pkl")    
        print("Saving full gradient dictionary at: {}".format(final_path))        
        with open(final_path, "wb") as f:
            pickle.dump(full_grad_param_dict, f, pickle.HIGHEST_PROTOCOL)

        return

class CIFARSampler:
    """
    Wraps your diffusion model + scheduler for FIDEvaluation.
    Must implement `.sample(batch_size, cfg_scale, sample_steps)`
    """
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.inference_mode()
    def sample(self, batch_size, cfg_scale=None, sample_steps=None):
        if sample_steps is None:
            sample_steps = len(self.scheduler)  # Default to full steps
        # Subsample timesteps evenly (e.g., every k-th step)
        total_steps = len(self.scheduler)
        step_indices = torch.linspace(0, total_steps - 1, sample_steps, dtype=torch.long).tolist()[::-1]
        timesteps = [int(i) for i in step_indices]
        
        x = torch.randn(batch_size, 3, 32, 32, device=self.device)
        
        for t in timesteps:
            t_repeat = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            residual = self.model(x, t_repeat, return_dict=False)[0]
            step_output = self.scheduler.step(residual, t, x)
            x = step_output.prev_sample
        
        return x
    
if "__main__" == __name__:

    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=[
            "experiment_name",
            "gpu_id",
            "phase",
            "val_batchsize",
            "testing_epoch",
            "input_emb",
            "time_emb",
            "beta_schedule",
            "num_val",
            "emb_size",
            "seed",
            "base_grad_dict_dir",
            "grad_dict_dir",
            "full_grads_path",
            "ckpt_path",
            "num_gen_imgs",
            "img_folder",
            "z_folder",
            "t_folder"
        ],
        sequence_args_and_types=[
            ("block_channels", int),
        ],
    )

    # Set the seed.
    set_seed(args.seed)

    if args.testing_epoch == -1:
        args.testing_epoch = args.max_epochs - 1

    cifar10_example = CIFAR10Example(args)
    if args.phase == "train":
        cifar10_example.train(args)
        cifar10_example.test(args)

    if args.phase == "compute_full_gradients":
        cifar10_example.compute_full_gradients(args)
    if args.phase == "compute_mini_batch_gradients":
        cifar10_example.compute_mini_batch_gradients(args)
    if args.phase == "test":
        cifar10_example.test(args)

    upload_to_cloud(args, rclone_remote="UCFOneDrive")