import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import csv
from tqdm import tqdm
from dataset import Dataset
import torch
import torch.nn as nn
from torchvision import models
import argparse
import pandas as pd
from albumentations import RandomRotate90,Resize, Flip, Normalize
from albumentations.core.composition import Compose, OneOf
from pyxconv.nvidia_mem_tracker import MemoryTracker
from train import create_model
import torch.optim as optim
import losses
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
from glob import glob


"""
Compute peak memory of UNeXT model for different batch sizes, probing vectors, and image dimensions 
via the new memory tracker.

Usage:
    sh bash_scripts/bash_compute_peak_memory.sh
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute peak memory of UNeXT model via the new memory tracker.'
    )
    parser.add_argument('--in_channel_size', type=int, default=1, help='Number of input channels.')
    parser.add_argument(
        '--mem_log_dir', 
        type=str, 
        required=True,
        help='Path to save the final memory heatmap'
    )
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment Variation.')
    parser.add_argument('--batch_size', type=int, default=14, help='Batch size')
    parser.add_argument('--probing_vector', type=str, help='Probing vector')
    parser.add_argument('--img_dim', type=int, required=True, help='Image dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset')
    parser.add_argument('--loss', default='BCEDiceLoss',
                    choices=LOSS_NAMES,
                    help='loss: ' +
                    ' | '.join(LOSS_NAMES) +
                    ' (default: BCEDiceLoss)')

    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    args = parser.parse_args()

    return args

def compute_iteration_memory(
    model, 
    train_loader, 
    device, 
    criterion, 
    optimizer,
    img_dim,
    in_channel_size
):
    """
    Compute the peak memory for one iteration of the model.

    Args:
        model: The model to compute the peak memory for.
        train_loader: The training loader.
        device: The device to compute the peak memory on.
        criterion: The criterion to compute the loss.
        optimizer: The optimizer to compute the peak memory for.
        img_dim: The dimension of the input image.
        in_channel_size: The number of input channels.
    Returns:
        peak_mem: The peak memory for one iteration of the model.
    """
    model.train()
    running_loss = 0.0

    for i, (images, target, _) in tqdm(enumerate(train_loader)):

        # images: (B, C, H, W) eg. (128, 3, 256, 256)
        # target: (B, H, W, C) eg. (128, 1, 256, 256)

        if i > 1:
            break

        B = images.shape[0]

        images = images.to(device)

        target = target.to(device)


        # Forward pass
        # (B, num_c) eg. (128, 10)
        with MemoryTracker() as t:

            # (B, num_classes, H, W) eg. (128, 1, 256, 256)
            outputs = model(images)

            # Backward and optimize
            optimizer.zero_grad()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        peak_mem = t.torch_peak/2**20

    return peak_mem

def main(args):
    model_name = args.model_name
    img_dim = args.img_dim
    in_channel_size = args.in_channel_size
    num_classes = args.num_classes
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    probing_vector = args.probing_vector
    exp_name = args.exp_name
    mem_log_dir = args.mem_log_dir
    dataset = args.dataset
    loss = args.loss
    num_epochs = 1

    # Use the custom GLAS dataset
    train_img_ids = sorted(glob(os.path.join(dataset, 'train', 'images', '*')))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(img_dim, img_dim),
        Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(dataset, 'train','images'),
        mask_dir=os.path.join(dataset, 'train','masks'),
        img_ext='.bmp',
        mask_ext='.bmp',
        num_classes=num_classes,
        transform=train_transform
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_sizes = [batch_size]
    probing_vectors = [probing_vector]

    # define loss function (criterion)
    if loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[args.loss]().to(device)
    rows = []

    # Store probing_vector for file naming (will be set in loop)
    final_probing_vector = None

    # Main training loop
    for batch_size in batch_sizes:
        print("Processing batch-size: {}".format(batch_size))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        
        for probing_vector in probing_vectors:
            base = (probing_vector == 'base')
            probing_vector_str = probing_vector  # Keep as string for create_model
            probing_vector_int = int(probing_vector) if not base else 'base'  # Convert to int for logging
            final_probing_vector = probing_vector_int  # Store for file naming
            print("Processing probing_vector: {}".format(probing_vector_str))

            model = create_model(
                num_classes=args.num_classes, 
                probing_vector=probing_vector_str,
                arch=model_name
            )
            params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.Adam(
                params, lr=args.lr, weight_decay=args.weight_decay
            )
            model.to(device)

            for _ in tqdm(range(num_epochs)):

                peak_mem = compute_iteration_memory(
                    model = model,
                    train_loader = train_loader,
                    device = device,
                    criterion = criterion,
                    optimizer = optimizer,
                    img_dim = img_dim,
                    in_channel_size = in_channel_size
                )
                print("Peak memory: {}".format(peak_mem))
                
                rows.append({
                    "batch_size": batch_size,
                    "probing_vector": probing_vector_int,
                    "peak_memory": peak_mem
                })
                
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    mem_bs_dir = "{}/{}_batch".format(mem_log_dir, batch_size)   
    if not os.path.exists(mem_bs_dir):
        os.makedirs(mem_bs_dir)     
    memory_csv_file = "{}/{}_img_dim_{}_batch_size_{}_probing_vector_{}_peak_memory.csv".format(
        mem_bs_dir, exp_name, img_dim, batch_size, final_probing_vector)
    df = pd.DataFrame(rows)
    df.to_csv(
        memory_csv_file,
        columns = ['batch_size', 'probing_vector', 'peak_memory'],
        index = False
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)