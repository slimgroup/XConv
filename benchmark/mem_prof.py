import torch
import torch.nn as nn
import nvidia_smi

from pyxconv import *
import sys


class Model(nn.Module):
    def __init__(self, N, mode, cc, ps=8):
        super(Model, self).__init__()
        self.N = N
        self.conv =  Xconv2D(cc, cc, (3, 3), bias=False, ps=ps, stride=1, padding=1)
        if mode == "true":
            self.conv = nn.Conv2d(cc, cc, (3, 3), bias=False, padding=1, stride=1)
        
    def forward(self, x):
        out = self.conv(x)
        for l in range(1, self.N):
            out = self.conv(out)
        return out

    def extra_repr(self):
        return f"Number of layer: {self.N}"


if __name__ == "__main__":
    cc = 16
    b = 128
    device = torch.device("cuda")
    try:
        mode = int(sys.argv[1])
    except:
        mode = 1
    mode = "probe" if mode == 1 else "true"

    c = Model(4, mode, cc).to(device)

    print("Network", c)
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    X = torch.randn(b, cc, 256, 256, device=device)
    print("Input size", X.shape)
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'GPU usage {mode} before forward: mem: {100 * (mem.used / mem.total):.3f}%, abs-mem: {mem.used / (1024**3)} (GiB)')      
    y2 = c(X).mean()
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'GPU usage {mode} after forward: mem: {100 * (mem.used / mem.total):.3f}%, abs-mem: {mem.used / (1024**3)} (GiB)') 
    y2.backward()
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'GPU usage {mode} after backward: mem: {100 * (mem.used / mem.total):.3f}%, abs-mem: {mem.used / (1024**3)} (GiB)') 
