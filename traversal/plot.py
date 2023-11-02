import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Circle

OFFSET = 0.05


def show_slide(figsize, im, pos_df, spot_radius, fig_labels):
    plt.ion()
    fig, (ax1, ax) = plt.subplots(nrows=1, ncols=2, figsize=(2 * figsize, figsize))

    ax1.imshow(im)
    ax1.set_axis_off()
    ax1.text(
        OFFSET / 2,
        1 - OFFSET,
        fig_labels[0],
        transform=ax1.transAxes,
        size=20,
        weight="bold",
    )

    ax.imshow(im)
    ax.set_axis_off()
    ax.text(
        OFFSET / 2,
        1 - OFFSET,
        fig_labels[1],
        transform=ax.transAxes,
        size=20,
        weight="bold",
    )
    circs = PatchCollection(
        [Circle((x, y), spot_radius) for x, y in pos_df[["x", "y"]].to_numpy()],
        picker=True,
    )
    n = len(pos_df)
    facecolors = np.asarray([list(to_rgba("lightgray"))] * n)
    edgecolors = np.asarray([list(to_rgba("darkgray"))] * n)
    alphas = np.full(n, 0.85)
    circs.set_facecolor(facecolors)
    circs.set_edgecolor(edgecolors)
    circs.set_alpha(alphas)
    ax.add_collection(circs)
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax, circs, facecolors, edgecolors


def pick_idx(fig):
    selected_idx = None

    def onpick(event):
        nonlocal selected_idx
        idx = event.ind[0]
        selected_idx = idx

    cid = fig.canvas.mpl_connect("pick_event", onpick)
    while selected_idx is None:
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)
    return selected_idx


LABEL_DIRS = ["top", "bottom", "left", "right"]


def dir_to_args(label, xy):
    x = xy[0]
    y = xy[1]
    if label == "top":
        y -= 10
        ha = "center"
        va = "bottom"
    elif label == "bottom":
        y += 10
        ha = "center"
        va = "top"
    elif label == "left":
        x -= 10
        ha = "right"
        va = "center"
    elif label == "right":
        x += 10
        ha = "left"
        va = "center"
    else:
        raise ValueError()
    return (x, y), ha, va


def draw_start(ax, pos_df, start_idx, start_label, circs, edgecolors, fontsize=10):
    edgecolors[start_idx] = to_rgba("black")
    if start_label is not None:
        xy, ha, va = dir_to_args(start_label, pos_df.loc[start_idx, ["x", "y"]])
        ax.annotate(
            "Start",
            xy,
            color="black",
            ha=ha,
            va=va,
            size=fontsize,
            weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        )
    circs.set_edgecolor(edgecolors)
    # plt.pause(1)


def draw_path(ax, pos_df, path_idxs, end_label, circs, edgecolors, fontsize=10):
    # edgecolor for path
    count = 1
    for path_idx in path_idxs[1:]:  # last index includes end_idx
        edgecolors[path_idx] = to_rgba("black")
        if path_idx == path_idxs[-1] and end_label is not None:
            xy, ha, va = dir_to_args(end_label, pos_df.loc[path_idx, ["x", "y"]])
            ax.annotate(
                "End",
                xy,
                color="black",
                ha=ha,
                va=va,
                size=fontsize,
                weight="bold",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
            )
        count += 1
    circs.set_edgecolor(edgecolors)
    # plt.pause(1)


def draw_clusters(
    fig,
    ax,
    path_idxs,
    clusters,
    circs,
    facecolors,
    edgecolors=None,
    cm_name="gist_rainbow",
    show_cbar=True,
    discrete_cm=False,
    fontsize=10,
):
    cmap = colormaps[cm_name]
    if discrete_cm:
        cmap_interp = np.arange(len(path_idxs))
    else:
        cmap_interp = np.linspace(0, 1, len(path_idxs))

    idxs = set(path_idxs)
    for idx, i in enumerate(clusters):
        if i != -1:
            facecolors[idx] = list(cmap(cmap_interp[i]))
            if edgecolors is not None and idx not in idxs:
                edgecolors[idx] = list(cmap(cmap_interp[i]))
    circs.set_facecolor(facecolors)
    if edgecolors is not None:
        circs.set_edgecolor(edgecolors)

    if show_cbar:
        bbox = ax.get_position()
        bounds = bbox.get_points()
        width = bounds[1, 0] - bounds[0, 0]
        height = bounds[1, 1] - bounds[0, 1]
        bounds[0, 0] += width / 10  # xmin
        bounds[0, 1] += height / 100 * 2.5  # ymin
        bounds[1, 0] -= width / 10  # xmax
        bounds[1, 1] -= height / 100 * 95  # ymax
        bbox.set_points(bounds)
        cax = plt.Axes(fig, bbox)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.add_axes(cax)

        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=len(path_idxs) - 1), cmap=cmap)
        fig.colorbar(sm, cax, orientation="horizontal")
        cax.set_axis_off()
        cax_bbox = cax.get_position()
        x_offset = cax_bbox.width * 0.01
        y_center = cax_bbox.ymax - cax_bbox.height / 2
        fig.text(
            cax_bbox.xmin - x_offset,
            y_center,
            "Start",
            weight="bold",
            ha="right",
            va="center",
            fontsize=fontsize,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        )
        fig.text(
            cax_bbox.xmax + x_offset,
            y_center,
            "End",
            weight="bold",
            ha="left",
            va="center",
            fontsize=fontsize,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
        )

    # plt.pause(1)
