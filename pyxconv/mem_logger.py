import torch
import nvidia_smi
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})

__all__ = ['log_mem', 'plot_mem']


def _get_gpu_mem(synchronize=True, empty_cache=True):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return mem.used


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if hook_type == 'pre':
            return
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1
        
        mem_all = _get_gpu_mem()
        torch.cuda.synchronize()
        lname = type(self).__name__
        lname = 'conv' if 'conv' in lname.lower() else lname
        lname = 'ReLU' if 'relu' in lname.lower() else lname
        
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': f"{lname}_{hook_type}",
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
        })

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)


def log_mem(model, inp, mem_log=None, exp=None):
    nvidia_smi.nvmlInit()
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = model(inp)
        loss = out.sum()
        loss.backward()
    except Exception as e:
        print(f"Errored with error {e}")
    finally:
        [h.remove() for h in hr]

    return mem_log


def plot_mem(
        df,
        exps=None,
        normalize_call_idx=False,
        normalize_mem_all=True,
        filter_fwd=False,
        return_df=False,
        output_file=None,
        name=None
):
    if exps is None:
        exps = df.exp.drop_duplicates()
    labels = {'std': 'Standard', 'relu': 'BitReLU', 'conv': 'Probed',
              'all': 'Probed+BitReLU'}
    fig, ax = plt.subplots(figsize=(10, 10))

    layer_list = []
    for exp in exps:
        df_ = df[df.exp == exp]
        print(exp, len(df_))

        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()

        if normalize_mem_all:
            mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = mem_all / 2 ** 30

        layer_idx = 0
        callidx_fwd = df_[(df_["layer_idx"] == layer_idx) and
                          (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
        if filter_fwd:    
            df_ = df_[df_["call_idx"] <= callidx_fwd]

        plot = df_.plot(ax=ax, y='call_idx', x='mem_all', ylabel='Layer',
                        xlabel='Memory (Gb)', label=labels[exp],
                        fontsize=20, linewidth=4)
        plot.axhline(y=callidx_fwd)
        if len(df_.layer_type) > len(layer_list):
            layer_list = list(df_.layer_type)
    
    ax.set_yticks(range(0, len(layer_list)))
    ax.set_ylim(len(layer_list), 0)
    ax.set_yticklabels(layer_list, fontsize=6, minor=False)
    
    if name:
        ax.set_title(name, fontsize=20)
        fig.tight_layout()
    if output_file:
        fig.savefig(f"{output_file}.pdf", bbox_inches="tight")

    if return_df:
        return df_
