from argparse import ArgumentParser
import os

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import torch
import torchvision.models as models
import pandas as pd

from pyxconv import convert_net, log_mem, plot_mem

def parse_args():
    parser = ArgumentParser(description="Memory benchmark of popular models")
    parser.add_argument(
        "--ps", default=16, type=int, help="Probing size (default is 0=no probing"
    )
    args = parser.parse_args()

    print(
        f"benchmarking memeory usage of standard networks with r={args.ps} probing vectors"
    )
    
    return args

def bench_mem(name, ps, mode, xmode, mem_log, rand_inp):

    model = getattr(models, name)()
    model.to("cuda")
    convert_net(model, "net", mode=mode, ps=ps, xmode=xmode)
    try:
        mem_log.extend(log_mem(model, rand_inp, exp=mode))
    except Exception as e:
        print(f"log_mem failed because of {e}")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main(args):
    
    bs = 4
    rand_inp = torch.rand(bs, 3, 224, 224).cuda()


    for net in ["squeezenet1_0", "squeezenet1_1", "resnet18", "resnet50"]:
        mem_log = []
        for mode in ["std", "all"]:
            bench_mem(
                name=net, 
                ps = args.ps, 
                mode=mode, 
                mem_log=mem_log,
                rand_inp=rand_inp,
                xmode='independent'
            )

        df = pd.DataFrame(mem_log)

        base_dir = "with_env_independent"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        output_file = f"{base_dir}/{net}"
        plot_mem(
            df,
            name=f"{net}, Input size: {rand_inp.shape}",
            output_file=f"{base_dir}/{net}",
        )
        print("Saving {}".format(output_file))

if __name__ == "__main__":
    args = parse_args()
    main(args)