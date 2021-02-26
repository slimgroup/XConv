import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import nvidia_smi

from pyxconv import *
import sys

#torch.set_deterministic(True)

if __name__ == "__main__":
    cc = 16
    b = 128
    device = torch.device("cuda")
    try:
        mode = int(sys.argv[1])
    except:
        mode = 1
    print(mode, mode==1)
    c = Xconv2D(cc, cc, (3, 3), bias=False, ps=32, stride=1, padding=1).to(device)
    if mode != 1:
        c = nn.Conv2d(cc, cc, (3, 3), bias=False, padding=1, stride=1).to(device)
    mode = "probe" if mode == 1 else "true"
    #c = Xconv3D(cc, cc, (3, 3, 3), bias=False, ps=32, stride=1, padding=1).to(device)
    #c2 = nn.Conv3d(cc, cc, (3, 3, 3), bias=False, padding=1, stride=1).to(device)


    c = nn.Sequential(c, c, c, c)
    print("Network", c)
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    X = torch.randn(b, cc, 256, 256, device=device)
    print("Input size", X.shape)
    #X = torch.randn(16, cc, 256, 256, 32, device=device)
    with profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True) as prof:
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f'GPU usage {mode} before forward: mem: {100 * (mem.used / mem.total):.3f}%, abs-mem: {mem.used / (1024**3)} (GiB)')      
        y2 = c(X).mean()
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f'GPU usage {mode} after forward: mem: {100 * (mem.used / mem.total):.3f}%, abs-mem: {mem.used / (1024**3)} (GiB)') 
        y2.backward()
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f'GPU usage {mode} after backward: mem: {100 * (mem.used / mem.total):.3f}%, abs-mem: {mem.used / (1024**3)} (GiB)') 
    print('\n \n \n')
    prof.export_chrome_trace(f"trace{mode}.json")
    #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
