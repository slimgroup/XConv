import os
from projorg import datadir
import pickle


class Dictlist(dict):
    def __setitem__(self, key, value):
        try:
            self[key]
        except KeyError:
            super(Dictlist, self).__setitem__(key, [])
        self[key].append(value)


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def create_cifar100_coarse_to_fine():
    x = unpickle(os.path.join(datadir("datasets"), "cifar-100-python/train"))
    fine_to_coarse = Dictlist()

    for i in range(0, len(x[b"coarse_labels"])):
        fine_to_coarse[x[b"coarse_labels"][i]] = x[b"fine_labels"][i]
    d = dict(fine_to_coarse)
    for i in d.keys():
        d[i] = list(dict.fromkeys(d[i]))
    return d
