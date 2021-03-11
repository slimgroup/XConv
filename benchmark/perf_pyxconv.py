import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import pickle

import matplotlib.pyplot as plt

from pyxconv import *


def do_conv(c, X):
    y = c(X).mean()
    grad = y.backward()

if __name__ == "__main__":
    num_threads = torch.get_num_threads()
    pss = [0, 2, 4, 8, 16, 32, 64]
    Ns = [2**i for i in range(5, 11)]
    bs = [2**i for i in range(5, 9)]
    cis = [2**i for i in range(0, 10, 3)]

    device = torch.device("cuda")
    results = []

    for ps in pss:
        for ci in cis:
            if ps > 0:
                cname = f'Xconv2D_{ps}'
                c = Xconv2D(ci, ci, (3, 3), bias=False, ps=ps, stride=1, padding=1).to(device)
            else:
                cname = 'nn.Conv2d'
                c = nn.Conv2d(ci, ci, (3, 3), bias=False, padding=1, stride=1).to(device)
            for b in bs:
                for n in Ns:
                   print(f"b={b}, N={n}x{n}, ci={ci}, ps={ps}")
                   try:
                       X = torch.randn(b, ci, n, n, device=device)
                       results.append(benchmark.Timer(stmt='do_conv(c, X)',
                           globals={'X': X, 'c': c, 'do_conv': do_conv},
                           label=f"2D conv b={b}", num_threads=num_threads,
                           sub_label=f"{n}x{n}, ci={ci}",
                           description=cname).blocked_autorange(min_run_time=1))
                   except:
                       pass
            with open('pybench.pickle', 'wb') as handle:
                pickle.dump(results, handle)
    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()
