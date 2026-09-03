# pylint: disable=E1102
# pylint: disable=invalid-name
"""ResNet18/34/50/101/152 in Pytorch.
Author: Lorenzo Luzi
"""
import torch
import torchvision
import os
from .resnet import ResNet18
RESNET_DIR_NAME = 'mnist_resnet18'


def get_model(f_model):
    #Load the model
    model = ResNet18(num_classes=10, init_channels=1, init_embedding=4)

    try:
        model.load_state_dict(torch.load(f_model, map_location='cpu'))
    except:
        model.load_state_dict(torch.load(f_model))

    model.eval()
    model.requires_grad_(False)
    return model


def get_feats(model, x):
    #Normalize features correctly
    x -= x.min()
    x /= x.max()  #Brings x to 0,1
    x = torchvision.transforms.Normalize((0.1307, ), (0.3081, ))(x)

    return model.forward_fe(x)


def sqrtm(matrix):
    #Make sure it isn't a single value
    if matrix.numel() == 1:
        return matrix.sqrt()

    #Get s and v
    s, v = torch.linalg.eigh(matrix)

    #Rectify the numerical errors
    s = torch.nn.ReLU()(s)

    # Get square root
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def get_mean(feats, f_save=None):
    #If no saving
    if f_save is None:
        return feats.mean(0)

    #If saving
    if os.path.exists(f_save):
        return torch.load(f_save)
    else:
        os.makedirs(os.path.dirname(f_save))
        mu = feats.mean(0)
        torch.save(mu, f_save)
        return mu


def get_cov(feats, f_save=None):
    #If no saving
    if f_save is None:
        return (feats.T @ feats) / (feats.shape[0] - 1)

    #If saving
    if os.path.exists(f_save):
        return torch.load(f_save)
    else:
        cov = (feats.T @ feats) / (feats.shape[0] - 1)
        torch.save(cov, f_save)
        return cov


#Get FID (this is squared) using torch
def get_fid(x, y):
    #Get stats (treat lists like stats otherwise features
    if type(x) == list:
        m0 = x[0]
        Sigma0 = x[1]
    else:
        m0 = get_mean(x)
        Sigma0 = get_cov(x)
    if type(y) == list:
        m1 = y[0]
        Sigma1 = y[1]
    else:
        m1 = get_mean(y)
        Sigma1 = get_cov(y)

    #Use sqrtm, which is pytorch compatible.
    Sigma00 = sqrtm(Sigma0)
    Sigma010 = sqrtm(Sigma00 @ Sigma1 @ Sigma00)
    d = torch.norm(m0 - m1)**2 + torch.trace(Sigma0 + Sigma1 - 2 * Sigma010)
    d = torch.abs(
        d)  #Modification because sometimes d has small imaginary parts
    return d.item()


def compute_fid(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    fe_model: torch.nn.Module,
) -> float:
    """
    Compute FID for a given model.
    Args:
        x: Original images. (N_samples, C, H, W) eg. (60000, 1, 28, 28)
        x_hat: Generated images. (N_samples, C, H, W) eg. (60000, 1, 28, 28)
        fe_model: Feature extractor model. (ResNet18 model)
    Returns:
        FID value. (float)
    """
    # Validate input shapes
    if x.dim() != 4:
        raise ValueError(
            f"x must be of shape (N, C, H, W), but got shape {tuple(x.shape)} "
            f"with {x.dim()} dimensions. Expected 4 dimensions."
        )
    if x_hat.dim() != 4:
        raise ValueError(
            f"x_hat must be of shape (N, C, H, W), but got shape {tuple(x_hat.shape)} "
            f"with {x_hat.dim()} dimensions. Expected 4 dimensions."
        )
    
    #Get features
    # (N_samples, dim) eg. (60000, 32)
    feats_x = get_feats(fe_model, x)

    # (dim,) eg. (32,)
    mu_x = get_mean(feats_x)

    # (dim, dim) eg. (32, 32)
    Sigma_x = get_cov(feats_x)

    # (N_samples, dim) eg. (60000, 32)
    feats_x_hat = get_feats(fe_model, x_hat)

    #Get FID
    fid_val = get_fid([mu_x, Sigma_x], feats_x_hat)

    return fid_val
