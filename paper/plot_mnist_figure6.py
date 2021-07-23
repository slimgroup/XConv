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
ps_n = set()
for root, dirs, files in os.walk("./mnist_bench/", topdown=False):
    try:
        print(f"Reading {files[0]}")
        b, ps = tuple(int(i) for i in root.split('/')[-1].split('_'))
        b_sizes.update({b})
        ps_n.update({ps})
        all_log[b][ps] = tflog2pandas(os.path.join(root, files[0]))
    except:
        pass


def sort_l(t):
    print(t)
    maxv = t[0].split(',')[1].split('=')[1]
    return maxv


plt.figure(figsize=(12, 8))
for i, b in enumerate(sorted(b_sizes)):
    print(i, b)
    plt.subplot(3, 2, i+1)
    for p in ps_n:
        maxv = np.max(all_log[b][p]['test']['Accuracy/test'])
        lab = f"Multi r={p}, max={maxv:.4f}" if p > 0 else f"True, max={maxv:.4f}"
        plt.plot(all_log[b][p]['test']['Accuracy/test'], label=lab)
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles),
                                  key=lambda t: sort_l(t), reverse=True))
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.title(f"B={b}")
    plt.legend(handles, labels)
plt.tight_layout()
plt.savefig("Acc_mnist.png", bbox_inches="tight")
plt.show()
