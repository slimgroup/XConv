import pandas as pd
from matplotlib import pyplot as plt

import numpy as np
import os
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        tag_train = [t for t in tags if 'test' not in t]
        tag_test = [t for t in tags if 'test' in t]
        d_train = {t: [] for t in tag_train}
        d_test = {t: [] for t in tag_test}
        for tag in tag_train:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            d_train[tag] = values
       
        for tag in tag_test:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            d_test[tag] = values

        d_train = pd.DataFrame(d_train)
        d_test = pd.DataFrame(d_test)
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return {'train': d_train, 'test': d_test}


all_log = defaultdict(dict)
b_sizes = set()
for root, dirs, files in os.walk("./cifar", topdown=False):
    try:
        b, ps, mode = tuple(i for i in root.split('/')[-1].split('_'))
        print(f"Reading {files[0]}, (b, ps, mode) = ({b}, {ps}, {mode})")
        b_sizes.update({(b, ps, mode)})
        all_log[(b, ps, mode)] = tflog2pandas(os.path.join(root, files[0]))
    except:
        pass


def sort_l(t):
    maxv = t[0].split(',')[1].split('=')[1]
    return maxv


legends = {'ortho': 'Multi-Ortho', 'all': 'Multi', 'features': 'Indep.', 'True': 'True'}

plt.figure(figsize=(12, 8))

for b in b_sizes:
    i = 4
    for t, k in zip(['test', 'train', 'test', 'train'], ['Accuracy/test', 'Accuracy/train', 'Loss/test', 'Loss/train']):
        plt.subplot(2, 2, i)
        i-=1
        maxv = np.min(all_log[b][t][k]) if 'Loss' in k else np.max(all_log[b][t][k])
        m = "min" if 'Loss' in k else "max"
        print(m, k, 'Loss' in k)
        lab = f"r={b[1]}, mode={legends[b[2]]}, {m}={maxv:.4f}" if int(b[1]) > 0 else f"True, {m}={maxv:.4f}"
        vals = all_log[b][t][k]
        xt = np.linspace(0, 100, len(vals))
        plt.plot(xt, vals, label=lab)
        plt.xlabel('Epochs')
        plt.title("%s %s" % tuple(k.split('/')[::-1]))
        plt.legend()
plt.tight_layout()
plt.savefig("cifar10.png", bbox_inches="tight")
plt.show()
