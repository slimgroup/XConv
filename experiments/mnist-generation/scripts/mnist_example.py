# pylint: disable=E1102
# pylint: disable=invalid-name
import argparse
import os
import sys
import torch
import torchvision
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid

# Path to this file: sips/scripts/mnist_example.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the *outer* projorg directory: sips/projorg
PROJORG_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "projorg"))

# Prepend so this takes precedence over any site-packages projorg
if PROJORG_ROOT not in sys.path:
    sys.path.insert(0, PROJORG_ROOT)


from projorg import (
    checkpointsdir,
    datadir,
    plotsdir,
    setup_environment,
)
from sips.utils import plot_loss
from tqdm import tqdm
import numpy as np

def set_seed(seed: int) -> None:
    """Set the seed for the random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class MNISTExample:
    """A class demonstrating the MNIST example.

    Attributes:
        device (torch.device): The device (cpu/cuda) used for computation.
        score_model: The score neural network.
        optimizer: The optimizer used for training.
        train_obj: Placeholder for training objective values.
        val_obj: Placeholder for validation objective values.
    """

    def __init__(self, args: argparse.ArgumentParser) -> None:
        """
        Initialize the GaussianExample object.

        Args:
            args (argparse.ArgumentParser): The command line arguments.
        """
        # Setting default device (cpu/cuda) depending on CUDA availability and
        # input arguments.
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        # Obtain training samples.
        x_train = torchvision.datasets.MNIST(
            datadir("datasets"),
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (1.0)),
                ]
            ),
        )

        # Obtain validations samples.
        x_val = torchvision.datasets.MNIST(
            datadir("datasets"),
            train=False,
            download=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (1.0)),
                ]
            ),
        )

        # Setup the batch index generator.
        self.train_loader = torch.utils.data.DataLoader(
            x_train,
            batch_size=args.batchsize,
            shuffle=True,
            drop_last=True,
            num_workers=16,
        )
        self.val_loader = torch.utils.data.DataLoader(
            x_val,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=True,
            num_workers=16,
        )

        # Initialize the network that will learn the score function.
        self.score_model = UNet2DModel(
            in_channels=1,
            out_channels=1,
            sample_size=x_train.data.shape[1:],
            time_embedding_type=args.time_emb,
            attention_head_dim=args.attn_dim,
            block_out_channels=args.block_channels,
            layers_per_block=args.block_nlayers,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        ).to(self.device)

        # Forward diffusion process noise scheduler.
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.nt,
            beta_schedule=args.beta_schedule,
        )

        # Optimization subroutine.
        self.optimizer = torch.optim.AdamW(
            self.score_model.parameters(),
            lr=args.lr,
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_epochs * len(self.train_loader),
        )

        # Some placeholders.
        self.train_obj = []
        self.val_obj = []

    def load_checkpoint(
        self, 
        args: argparse.ArgumentParser,
        ckpt_path: str = None
    ) -> None:
        """Load model checkpoint.

        Args:
            args (argparse.ArgumentParser): The command line arguments.
            ckpt_path (str): The path to the checkpoint.
            If None, the checkpoint is loaded from the experiment directory.
            (args.experiment)
        Raises:
            ValueError: If checkpoint does not exist or if filename and loaded
            checkpoint epoch are inconsistent.
        """
        if ckpt_path is None:
            file_to_load = os.path.join(
                checkpointsdir(args.experiment),
                'checkpoint_' + str(args.testing_epoch) + '.pth',
            )
        else:
            file_to_load = ckpt_path

        print(f"Loading checkpoint from {file_to_load}")

        if os.path.isfile(file_to_load):
            if self.device == torch.device(type="cpu"):
                checkpoint = torch.load(
                    file_to_load, map_location="cpu", weights_only=False
                )
            else:
                checkpoint = torch.load(file_to_load, weights_only=False)

            self.score_model.load_state_dict(checkpoint["model_state_dict"])

            self.train_obj = checkpoint["train_obj"]
            self.val_obj = checkpoint["val_obj"]

            if not args.testing_epoch == checkpoint["epoch"]:
                raise ValueError("Inconsistent filename and loaded checkpoint.")
        else:
            raise ValueError("Checkpoint does not exist.")

    def train(self, args: argparse.ArgumentParser) -> None:
        """Trains the hypernetwork.

        Args:
            args (argparse.ArgumentParser): The command line arguments.
        """

        for epoch in tqdm(
            range(args.max_epochs),
            unit="epoch",
            colour="#B5F2A9",
            dynamic_ncols=True,
            desc="Training progress",
        ):
            # Validation phase.
            self.score_model.eval()

            with torch.no_grad():
                self.val_obj.append(0.0)

                for x_val, _ in self.val_loader:
                    x_val = x_val.to(self.device)

                    noise = torch.randn(
                        x_val.shape,
                        device=self.device,
                    )

                    timesteps = torch.randint(
                        0,
                        len(self.noise_scheduler),
                        (args.batchsize,),
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

                # Average and store validation objective.
                self.val_obj[-1] = self.val_obj[-1].item() / len(
                    self.val_loader
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
                        (args.batchsize,),
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

                    # Update parameters.
                    self.optimizer.step()
                    self.optimizer.zero_grad()

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
            if epoch % 10 == 0:
                self.test(args, epoch=epoch)

            # Save model checkpoints.
            if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                torch.save(
                    {
                        "model_state_dict": self.score_model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "train_obj": self.train_obj,
                        "val_obj": self.val_obj,
                    },
                    os.path.join(
                        checkpointsdir(args.experiment),
                        "checkpoint_" + str(epoch) + ".pth",
                    ),
                )

    def test(self, args: argparse.ArgumentParser, epoch=-1) -> None:
        """Performs testing and plots the results.

        Args:
            args (argparse.ArgumentParser): The command line arguments.
            epoch (int): The current epoch.

        Returns:
            None
        """
        if epoch == -1:
            # Load the network from the checkpoint.
            self.load_checkpoint(args)
            epoch = args.testing_epoch

        # Set the hypernetwork to evaluation mode.
        self.score_model.eval()

        # Create a pipeline for generating images.
        pipeline = DDPMPipeline(
            unet=self.score_model,
            scheduler=self.noise_scheduler,
        )

        # Generate images.
        images = pipeline(batch_size=16).images

        # Save the images.
        image_grid = make_image_grid(images[:16], rows=4, cols=4)
        image_grid.save(
            f"{os.path.join(plotsdir(args.experiment))}/{epoch:04d}.png"
        )

        # Plot the training and validation loss.
        plot_loss(
            args, 
            self.train_obj, 
            self.val_obj, 
            epoch=epoch
        )

def parser_user_args() -> argparse.ArgumentParser:
    """Parse the user arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", 
        type=str, 
        required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        required=True,
        help="Random seed (overrides config file). If not provided, the seed from the config file is used."
    )
    parser.add_argument(
        "--run_num", 
        type=int, 
        required=True,
        help="Run number."
    )
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    # Read input arguments from a json file and make an experiment name.
    
    
    user_args = parser_user_args()
    config_file = user_args.config_file
    seed = user_args.seed
    run_num = user_args.run_num

    # Remove --config_file and --seed, --run_num from sys.argv so setup_environment doesn't try to parse them
    for arg_name in ['--config_file', '--seed', '--run_num']:
        if arg_name in sys.argv:
            idx = sys.argv.index(arg_name)
            # Remove both the argument and its value
            sys.argv.pop(idx)
            if idx < len(sys.argv) and not sys.argv[idx].startswith('--'):
                sys.argv.pop(idx)

    args = setup_environment(
        config_file = config_file,
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
            "config_file"
        ],
        sequence_args_and_types=[
            ("block_channels", int),
        ],
        user_args=user_args,
    )

    # Set the seed.
    set_seed(args.seed)

    if args.testing_epoch == -1:
        args.testing_epoch = args.max_epochs - 1

    print("Saving experiment information to " + args.experiment)

    mnist_example = MNISTExample(args)
    if args.phase == "train":
        mnist_example.train(args)

    mnist_example.test(args, args.testing_epoch)