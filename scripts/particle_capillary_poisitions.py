import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import argparse

cividis_null = plt.get_cmap("cividis")
cividis_null.set_bad((0.0, 0.0, 0.0))


def plot_capillary_heatmap(particles: np.ndarray, resolution_um: int = 25):
    hist, _, _ = np.histogram2d(
        particles["y"],
        particles["x"],
        bins=(
            np.arange(0, 1536, resolution_um),
            np.arange(0, 2048, resolution_um),
        ),
    )
    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("particles", help="path to partciles.csv")
    parser.add_argument(
        "--resolution", type=float, default=25.0, help="resolution of plot"
    )

    args = parser.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[3, 1])
    axes[0].set_axis_off()

    data = np.genfromtxt(args.particles, names=True, delimiter=",")
    im = axes[0].imshow(
        plot_capillary_heatmap(data, args.resolution), cmap=cividis_null, norm="log"
    )

    axes[1].hist(data["radius"] * 2.0 * 0.46, bins=np.arange(0.0, 30.0, 0.1))
    axes[1].set_xlabel("Size (µm)")

    def update(xmin: float, xmax: float):
        x = data[data["radius"] * 2.0 * 0.46 > xmin]
        x = x[x["radius"] * 2.0 * 0.46 < xmax]

        im.set_data(plot_capillary_heatmap(x, args.resolution))
        fig.canvas.draw_idle()

    span = SpanSelector(
        axes[1],
        update,
        "horizontal",
        useblit=True,
        interactive=True,
        props=dict(alpha=0.25, facecolor="tab:red"),
        drag_from_anywhere=True,
    )
    span.extents = (0.0, 30.0)
    span.set_active(True)

    plt.show()
