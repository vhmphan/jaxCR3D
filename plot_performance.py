from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent

performance = np.loadtxt(
    folder / "performance_js.csv", skiprows=1, ndmin=2
)
reference_time = np.loadtxt(folder / "digitized_timings.txt")
reference_memory = np.loadtxt(folder / "digitized_memories.txt")

# Unit of the digitized memory figure: confirm against its original label.
reference_memory_unit = "GiB"
to_gib = {"GiB": 1.0, "GB": 1e9 / 1024**3, "MiB": 1 / 1024}

fig, axes = plt.subplots(
    1, 2, figsize=(14, 5), constrained_layout=True
)

labels = [
    "D-Solve", "D-Forward", "D-Reverse",
    "J-Solve", "J-Forward", "J-Reverse",
]
colors = ["black", "crimson", "tab:blue"] * 2
markers = ["o", ">", "<"] * 2

my_labels = ["My solve", "My JVP", "My VJP"]
my_colors = ["forestgreen", "darkorange", "purple"]

for ax, reference, measured, factor, title, ylabel in [
    (
        axes[0], reference_time, performance[:, 1:4],
        1.0, "Runtime", "Time [s]",
    ),
    (
        axes[1], reference_memory, performance[:, 4:7],
        to_gib[reference_memory_unit], "Memory", "Memory [GiB]",
    ),
]:
    # Six digitized reference curves.
    for i, (label, color, marker) in enumerate(
        zip(labels, colors, markers)
    ):
        ax.plot(
            reference[:, 1], reference[:, i + 2] * factor,
            color=color, marker=marker,
            linestyle="-" if i < 3 else "--",
            label=label,
        )

    # Your three measured curves.
    for i, (label, color) in enumerate(zip(my_labels, my_colors)):
        ax.plot(
            2.0**performance[:, 0], measured[:, i],
            "s-", color=color, label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Reference ticks, plus the largest measured size.
    ticks = np.unique(
        np.append(reference[:, 0], performance[:, 0].max())
    )
    ax.set_xticks(
        2.0**ticks, [rf"$2^{{{s:g}}}$" for s in ticks]
    )

    ax.set(
        xlabel="Number of grid points",
        ylabel=ylabel,
        title=title,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)

fig.savefig(folder / "fg_performance_js.png", dpi=200)


from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
cg = np.loadtxt(folder / 'performance.csv', skiprows=1, ndmin=2)
js = np.loadtxt(folder / 'performance_js.csv', skiprows=1, ndmin=2)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
colors = ['black', 'tab:red', 'tab:blue']
operations = ['Solve', 'JVP', 'VJP']
for ax, offset, title, ylabel in [
    (axes[0], 1, 'Runtime', 'Time [s]'),
    (axes[1], 4, 'Memory', 'Estimated memory [GiB]'),
]:
    for data, name, style, marker in [(cg, 'CG', '-', 'o'), (js, 'JS', '--', 's')]:
        for i, (op, color) in enumerate(zip(operations, colors)):
            ax.plot(2.0**data[:, 0], data[:, offset+i],
                    linestyle=style, marker=marker, color=color,
                    markerfacecolor='none' if op == 'VJP' else color,
                    markersize=8 if op == 'VJP' else 5,
                    label=f'{name} {op}')
    ticks = np.unique(np.concatenate((cg[:, 0], js[:, 0])))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(2.0**ticks, [rf'$2^{{{s:g}}}$' for s in ticks])
    ax.set(xlabel='Grid size N', ylabel=ylabel, title=title)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(fontsize=8)
fig.savefig(folder / 'fg_compare.png', dpi=200)
