import argparse
from pathlib import Path
import numpy as np
from src.solver import solve_u, sheet_source, residual_norm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle

def add_mask_circle(ax, Lr, zoom=False):
    """XY boundary (z=0) of the spherical data mask in example.py."""
    ax.add_patch(Circle(
        (-8.0, 0.0), radius=1.25, fill=False,
        edgecolor="white", linewidth=1.5, linestyle="--",
        label="Data mask (z = 0)",
    ))
    ax.legend(loc="upper left", fontsize="small")
    if zoom:
        ax.set(xlim=(- 8.0 - Lr, -8.0 + Lr), ylim=(-Lr, Lr))

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-dir", type=Path, default=Path("run"))
    parser.add_argument("--vi-iteration", type=int, default=1)
    parser.add_argument("--samples", type=Path, default=Path("g_samples_dx_0p2.npy"))
    # parser.add_argument("--samples", type=Path, default=Path("run-10/g_samples_dx_0p2.npy"))
    parser.add_argument("--truth", type=Path, default=Path("data/mock_g_dx_0p2.npy"))
    parser.add_argument("--zoom", action="store_true", help="Show +/-2 kpc around the mask center (x=8, y=0 kpc).")
    args = parser.parse_args()

    samples_path = (
        args.run_dir
        / f"vi-samples-{args.vi_iteration}"
        / args.samples
    )

    g_samples = np.load(samples_path)
    # g_samples = np.load(args.samples)
    g_mean = np.mean(g_samples, axis=0)
    g_truth = np.load(args.truth)
    dg = np.abs(g_mean - g_truth) / g_truth
    dg[dg>10.0] = np.nan

    dx = 0.2
    nz = int(8.0/dx) + 1

    U_mean = solve_u(sheet_source(g_mean, nz, dx), dx)
    U_truth = solve_u(sheet_source(g_truth, nz, dx), dx)
    dU = np.array(np.abs(U_mean[1:-1, 1:-1, nz//2] - U_truth[1:-1, 1:-1, nz//2]) / U_truth[1:-1, 1:-1, nz//2])
    dU[dU>10.0] = np.nan

    # Interior coordinates, assuming the domain is centred at zero.
    nx, ny = g_truth.shape
    L = (nx + 1) * dx / 2.0
    x = np.linspace(-L, L, nx+2)[1:-1]
    y = np.linspace(-L, L, ny+2)[1:-1]

    output_dir = args.run_dir / f"vi-samples-{args.vi_iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.zoom:
        Lr = 2.0
        output = output_dir / "fg_posterior_zoom.png"
        output_samp = output_dir / "fg_samples_zoom.png"
    else:
        Lr = x[-2]
        output = output_dir / "fg_posterior.png"
        output_samp = output_dir / "fg_samples.png"

    U_truth_xy = np.asarray(U_truth[1:-1, 1:-1, nz//2])
    U_mean_xy = np.asarray(U_mean[1:-1, 1:-1, nz//2])

    fields = (
        (g_truth, g_mean, dg),
        (U_truth_xy, U_mean_xy, dU),
    )
    titles = (
        ("Ground truth g", "Posterior mean g", "Relative error dg"),
        ("Ground truth U at z = 0", "Posterior mean U at z = 0",
        "Relative error dU"),
    )

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9), constrained_layout=True
    )

    for row in range(2):
        # Use the same colour scale for the truth and mean in each row.
        scale = np.nanmean(fields[row][0])
        shared_norm = LogNorm(vmin=scale / 5, vmax=scale * 5)

        for col in range(3):
            ax = axes[row, col]
            a = np.asarray(fields[row][col])
            norm = shared_norm
            if col == 2:
                scale = np.nanmean(a)
                norm = LogNorm(vmin=scale / 100, vmax=10.0) # scale * 100)

            im = ax.pcolormesh(
                x, y, a.T,
                shading="auto",
                cmap="magma",
                norm=norm,
            )

            ax.set(
                xlabel="x [kpc]",
                ylabel="y [kpc]",
                title=titles[row][col],
                aspect="equal"
            )

            add_mask_circle(ax, Lr, zoom=args.zoom)

            label = "Relative error" if col == 2 else ("g" if row == 0 else "U")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, label=label)

    fig.savefig(output, dpi=150)
    plt.close(fig)

    fields = (
        (g_truth, g_samples[0], g_samples[1]),
        (None,    g_samples[2], g_samples[3]),
    )
    titles = (
        ("Ground truth", "Sample 0", "Sample 1"),
        ("",            "Sample 2", "Sample 3"),
    )

    scale = np.nanmean(g_truth)
    norm = LogNorm(vmin=scale / 5, vmax=scale * 5)

    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9), constrained_layout=True
    )

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            a = fields[row][col]

            if a is None:
                ax.set_axis_off()
                continue

            im = ax.pcolormesh(
                x, y, np.asarray(a).T,
                shading="auto",
                cmap="magma",
                norm=norm,
            )

            ax.set(
                xlabel="x [kpc]",
                ylabel="y [kpc]",
                title=titles[row][col],
                aspect="equal",
            )

            add_mask_circle(ax, Lr, zoom=args.zoom)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, label="g")

    fig.savefig(output_samp, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
