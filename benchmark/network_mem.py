# %% Imports
import pandas as pd
import torch
from torch import nn
from torchvision.models import alexnet, vgg16, squeezenet1_1, resnet18

from pyxconv import convert_net, Xconv2D, log_mem, plot_mem

# %% Analysis baseline
mode = resnet18

model = mode()

bs = 128
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

convert_net(model, 'net')
try:
    mem_log.extend(log_mem(model, input, exp='relu'))
except Exception as e:
    print(f'log_mem failed because of {e}')

torch.cuda.synchronize()
torch.cuda.empty_cache()

df = pd.DataFrame(mem_log)

base_dir = '.'
nname = ('%s'%type(model)).split('.')[-1].split("'")[0]
plot_mem(df, name=f"{nname}, Input size: {input.shape}", output_file=f'{base_dir}/{nname}')

