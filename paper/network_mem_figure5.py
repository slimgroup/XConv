from argparse import ArgumentParser
import os

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch
import torchvision.models as models
import pandas as pd

from pyxconv import convert_net, log_mem, plot_mem

parser = ArgumentParser(description="Memory benchmark of popular models")
parser.add_argument(
    "--ps", default=16, type=int, help="Probing size (default is 0=no probing"
)
args = parser.parse_args()

print(
    f"benchmarking memeory usage of standard networks with r={args.ps} probing vectors"
)

bs = 4
input = torch.rand(bs, 3, 224, 224).cuda()


def bench_mem(name, ps, mode, mem_log):
    model = getattr(models, name)()
    model.to("cuda")
    convert_net(model, "net", mode=mode, ps=args.ps, xmode="gaussian")
    try:
        mem_log.extend(log_mem(model, input, exp=mode))
    except Exception as e:
        print(f"log_mem failed because of {e}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


for net in ["squeezenet1_0", "squeezenet1_1", "resnet18", "resnet50"]:
    mem_log = []
    for mode in ["std", "all"]:
        bench_mem(net, args.ps, mode, mem_log)

    df = pd.DataFrame(mem_log)

    base_dir = "."
    plot_mem(
        df,
        name=f"{net}, Input size: {input.shape}",
        output_file=f"{base_dir}/{net}",
    )
