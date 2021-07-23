import pickle
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.rcParams['font.size'] = 18

try:
    results = pickle.load(open("pybench.pickle", "rb"))
except:
    FileNotFoundError("Result file not found, run gpu_bench.py first")

bi = [32, 64, 128, 256]
ci = [3, 64, 128, 256]
ps = [32, 128, 256]

def as_plot(dict_in):
    return zip(*sorted(dict_in.items()))


t_times = defaultdict(dict)
p_times = {32: defaultdict(dict), 128: defaultdict(dict), 256: defaultdict(dict)}

for r in results:
    b_size = int(r.title.split('b=')[1].split(':')[0])
    n = int(r.title.split('x')[1].split(',')[0])
    c = int(r.title.split('ci=')[1])

    typec = 'torch' if r.description.startswith('nn') else 'Probed'
    np = 0 if typec == 'torch' else int(r.description.split('_')[1])

    to_fill = t_times if np == 0 else p_times[np]

    if b_size not in to_fill:
        to_fill[b_size] = defaultdict(dict)
    to_fill[b_size][c][n] = r.mean


col_p = ["--^m", "--^b", "--^c", "--^y", "--^g", "--^k"]

for b in bi:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True)
    fig.suptitle("  ") 
    for c, ax in zip(ci, axs.reshape(-1)):
        ax.loglog(*as_plot(t_times[b][c]), color='red', linestyle='--',
                  marker='o', label="True", basex=2, basey=10)
        for j,p in enumerate(ps):
            ax.loglog(*as_plot(p_times[p][b][c]), col_p[j], label=f"r={p}",
                      basex=2, basey=10)
        ax.set_xlabel('N (image is NxN)')
        ax.set_ylabel('Runtime (s)')
        ax.set_title(r"$C_{in}=C_{out}=$"+f"{c}")
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels,loc='upper center', bbox_to_anchor=(0.25, 0.52, 0.5, 0.5), ncol=4)
    plt.savefig(f"gpu_perf_{b}.png", bbox_inches="tight")

plt.show()

