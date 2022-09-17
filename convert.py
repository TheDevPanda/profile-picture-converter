import sys
import numpy as np
import click
from skimage import data, segmentation, color, filters, io
from skimage.transform import resize
from skimage.future import graph
from matplotlib import pyplot as plt
from itertools import product
from PIL import Image


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {"weight": 0.0, "count": 0}

    count_src = graph[src].get(n, default)["count"]
    count_dst = graph[dst].get(n, default)["count"]

    weight_src = graph[src].get(n, default)["weight"]
    weight_dst = graph[dst].get(n, default)["weight"]

    count = count_src + count_dst
    return {
        "count": count,
        "weight": (count_src * weight_src + count_dst * weight_dst) / count,
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


@click.command()
@click.option("-p", "--path", help="Path to image file")
@click.option("-g", "--grid", default=True, type=bool, help="Generate a parameter grid")
@click.option("-c", "--compactness", default=30, type=int)
@click.option("-s", "--segments", default=400, type=int)
@click.option("-t", "--thresh", default=0.08, type=float)
def convert(path, grid, compactness, segments, thresh):
    if grid:
        convert_grid(path)
    else:
        img = _convert(path, compactness, segments, thresh)
        show(img)
        save(img, "out.png")


def _convert(path, compactness, segments, thresh):
    img = io.imread(path)
    img = resize(img, (400, 400))
    edges = filters.sobel(color.rgb2gray(img))
    labels = segmentation.slic(
        img, compactness=compactness, n_segments=segments, start_label=1
    )
    g = graph.rag_boundary(labels, edges)

    # graph.show_rag(labels, g, img)
    # plt.title('Initial RAG')

    labels2 = graph.merge_hierarchical(
        labels,
        g,
        thresh=thresh,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_boundary,
        weight_func=weight_boundary,
    )

    # graph.show_rag(labels, g, img)
    # plt.title('RAG after hierarchical merging')

    out = color.label2rgb(labels2, img, kind="avg", bg_label=0)

    return out


def show(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


def save(img, name):
    img = (img * 255).astype(np.uint8)
    im = Image.fromarray(img).convert("RGB")
    im.save(name)


def convert_grid(path):
    comp = (10, 30, 60)
    seg = (50, 100, 300)
    thr = (0.001, 0.005, 0.01)

    fig, axs = plt.subplots(
        nrows=len(comp),
        ncols=len(seg) * len(thr),
        figsize=(12, 8),
        subplot_kw={"xticks": [], "yticks": []},
    )

    for (thresh, segments, compactness), ax in zip(product(thr, seg, comp), axs.flat):
        img = _convert(path, compactness, segments, thresh)
        ax.imshow(img)
        ax.set_title(f"C{compactness}S{segments}T{thresh}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    convert()
