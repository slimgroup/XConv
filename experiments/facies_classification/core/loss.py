import torch
import torch.nn.functional as F

def cross_entropy(input, target, weight=None, ignore_index=255):
    '''
    Use 255 to fill empty values when padding or doing any augmentation operations
    like rotation. 
    '''
    target = torch.squeeze(target,dim=1)
    loss = F.cross_entropy(input, target, weight, reduction='sum',  ignore_index=255)
    return loss


def cross_entropy_mean(input, target, weight=None, ignore_index=255):
    '''
    Mean cross-entropy over all non-ignored elements (used for gradient-error benchmarks).
    '''
    target = torch.squeeze(target, dim=1)
    loss = F.cross_entropy(input, target, weight, reduction='mean', ignore_index=ignore_index)
    return loss
