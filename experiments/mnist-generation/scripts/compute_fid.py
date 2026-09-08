# pylint: disable=E1102
# pylint: disable=invalid-name
import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
from tqdm import tqdm

# Path to this file: sips/scripts/compute_fid.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the *outer* projorg directory: sips/projorg
PROJORG_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "projorg"))

# Prepend so this takes precedence over any site-packages projorg
if PROJORG_ROOT not in sys.path:
    sys.path.insert(0, PROJORG_ROOT)

from projorg import (
    checkpointsdir,
    datadir,
    setup_environment,
)

from mnist_example import MNISTExample
from model.resnet.fid import get_model, compute_fid
from diffusers import DDPMPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Compute FID for a given model.')
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--samples_per_batch", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--resnet_dir_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        required=True,
        help='Number of inference steps.'
    )
    return parser.parse_args()

def compute_model_fid(
    model_config_file: str,
    ckpt_path: str,
    num_samples: int,
    samples_per_batch: int,
    orig_imgs: torch.Tensor,
    fe_model: torch.nn.Module,
    num_inference_steps: int,
) -> float:

    """
    Compute FID for a given model.
    Args:
        model_config_file: Path to the model configuration file.
        ckpt_path: Path to the checkpoint file.
        num_samples: Number of samples to generate.
        samples_per_batch: Number of samples to generate per batch.
        orig_imgs: Original images.
        fe_model: Feature extractor model.
        num_inference_steps: Number of denoising inference steps.
    Returns:
        FID value.
    """

    # Load model and sample images to compute FID
    args = setup_environment(
        model_config_file,
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
        ],
        sequence_args_and_types=[
            ("block_channels", int),
        ],
        user_args=None
    )

    if args.testing_epoch == -1:
        args.testing_epoch = args.max_epochs - 1


    model = MNISTExample(args)

    model.load_checkpoint(
        args=args,
        ckpt_path=ckpt_path,
    )

    # Set the hypernetwork to evaluation mode.
    model.score_model.eval()

    # Create a pipeline for generating images.
    pipeline = DDPMPipeline(
        unet=model.score_model,
        scheduler=model.noise_scheduler,
    )

    # Generate images.
    sampled_batches = []
    n_loops = num_samples // samples_per_batch

    with torch.no_grad():   
        for _ in tqdm(range(n_loops), desc="Sampling images"):

            # List of [batch_size] PIL images of shape (28, 28, 1)
            out = pipeline(
                batch_size = samples_per_batch, 
                output_type="pt",
                num_inference_steps=num_inference_steps,
            )  # (B, C, H, W) on device

            # (B, C, H, W) on device
            sampled_images = out.images  # assumed in [0,1]

            # match preprocessing of orig_imgs (ToTensor + Normalize(0.5, 1.0))
            # sampled_images = (sampled_images - 0.5) / 1.0  # same as Normalize((0.5,), (1.0,))
            sampled_batches.append(sampled_images)   # keep on CPU for FID if fe_model is CPU
    
    # (N_samples, H, W, C) eg. (60000, 28, 28, 1)
    sampled_images = np.concatenate(sampled_batches, axis=0)
    sampled_images = torch.from_numpy(sampled_images)

    # (N_samples, C, H, W) eg. (60000, 1, 28, 28)
    sampled_images = sampled_images.permute(0, 3, 1, 2)
    
    # Compute FID
    model_fid = compute_fid(
        x = orig_imgs,
        x_hat = sampled_images,
        fe_model = fe_model,
    )

    return model_fid


if '__main__' == __name__:

    # Save original sys.argv before parsing
    original_argv = sys.argv.copy()
    
    # Parse the main script arguments
    main_args = parse_args()
    num_samples = main_args.num_samples
    samples_per_batch = main_args.samples_per_batch
    batch_size = main_args.batch_size
    num_workers = main_args.num_workers
    model_config_file = main_args.model_config_file
    ckpt_path = main_args.ckpt_path
    resnet_dir_name = main_args.resnet_dir_name
    model_name = main_args.model_name
    num_inference_steps = main_args.num_inference_steps
    
    # Restore sys.argv to only contain script name
    # This prevents setup_environment() from seeing the parsed arguments
    # (config file is passed as parameter, not command-line arg)
    sys.argv = [original_argv[0]]


    # Load original images.
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

    train_loader = torch.utils.data.DataLoader(
        x_train,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    
    # (N_samples, C, H, W) eg. (60000, 1, 28, 28)
    orig_imgs = torch.cat([batch[0] for batch in train_loader], dim=0)
    
    #Load feature extractor model.
    fe_model = get_model(
        os.path.join(
            checkpointsdir(resnet_dir_name),
            'mnist_resnet18.pt',
        ))

    curr_model_fid = compute_model_fid(
        model_config_file=model_config_file,
        ckpt_path=ckpt_path,
        num_samples=num_samples,
        samples_per_batch=samples_per_batch,
        orig_imgs=orig_imgs,
        fe_model=fe_model,
        num_inference_steps=num_inference_steps,
    )
    print(f'{model_name} Model FID:', curr_model_fid)
