import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import argparse


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

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    fig.subplots_adjust(bottom=0.2)

    data = np.genfromtxt(args.particles, names=True, delimiter=",")
    # ax.hist(data["radius"] * 2.0 * 0.46, bins=np.arange(0.0, 30.0, 0.1))
    ax.set_xlabel("Size (µm)")

    ax_aspect = fig.add_axes((0.1, 0.05, 0.8, 0.02))
    slider_aspect = RangeSlider(ax_aspect, "Aspect", 0.0, 1.0, valinit=(0.0, 1.0))

    ax_convex = fig.add_axes((0.1, 0.08, 0.8, 0.02))
    slider_convex = RangeSlider(ax_convex, "Circ.", 0.0, 1.0, valinit=(0.0, 1.0))

    ax_circ = fig.add_axes((0.1, 0.11, 0.8, 0.02))
    slider_circ = RangeSlider(ax_circ, "Convex.", 0.0, 1.0, valinit=(0.0, 1.0))

    # ax_aspect = fig.add_axes((0.1, 0.14, 0.8, 0.02))
    # slider_aspect = RangeSlider(ax_aspect, "Aspect", 0.0, 1.0, valinit=(0.0, 1.0), dragging=False)

    def update(unsused=None):
        x = data[
            (data["aspect"] >= slider_aspect.val[0])
            & (data["aspect"] <= slider_aspect.val[1])
        ]

        x = x[
            (x["convexity"] >= slider_convex.val[0])
            & (x["convexity"] <= slider_convex.val[1])
        ]
        x = x[
            (x["circularity"] >= slider_circ.val[0])
            & (x["circularity"] <= slider_circ.val[1])
        ]
        ax.clear()
        ax.hist(x["radius"] * 2.0 * 0.46, bins=np.arange(0.0, 30.0, 0.1))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_release_event", update)
    update()

    plt.show()
