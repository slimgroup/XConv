# %% Imports
import pandas as pd
import torch
from argparse import ArgumentParser
from torch import nn
import torchvision.models as models

from pyxconv import convert_net, Xconv2D, log_mem, plot_mem

parser = ArgumentParser(description="Memory benchmark of popular models")
parser.add_argument("-network", default="SqueezeNet",
                    help="Network ot benchmark")
args = parser.parse_args()

# %% Analysis baseline
try:
    mode = getattr(models, args.network)
except AttributeError:
    print(f"Network {args.network} not defined in torchvision.models, defaulting back to squeezenet1_1")
    mode = squeezenet1_1

print(f"benchmarking {args.network}")

model = mode()

bs = 24
input = torch.rand(bs, 3, 218, 178).cuda()
mem_log = []

model.to('cuda')

try:
    mem_log.extend(log_mem(model, input, exp='std'))
except Exception as e:
    print(f'log_mem failed because of {e}')

torch.cuda.synchronize()
torch.cuda.empty_cache()

convert_net(model, 'net', mode='conv')
try:
    mem_log.extend(log_mem(model, input, exp='conv'))
except Exception as e:
    print(f'log_mem failed because of {e}')

torch.cuda.synchronize()
torch.cuda.empty_cache()

model = mode().cuda()
convert_net(model, 'net', mode='relu')
try:
    mem_log.extend(log_mem(model, input, exp='relu'))
except Exception as e:
    print(f'log_mem failed because of {e}')

torch.cuda.synchronize()
torch.cuda.empty_cache()

convert_net(model, 'net')
try:
    mem_log.extend(log_mem(model, input, exp='all'))
except Exception as e:
    print(f'log_mem failed because of {e}')

torch.cuda.synchronize()
torch.cuda.empty_cache()

df = pd.DataFrame(mem_log)

base_dir = '.'
plot_mem(df, name=f"{args.network}, Input size: {input.shape}", output_file=f'{base_dir}/{args.network}')

