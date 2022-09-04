from skimage.future import graph
from skimage import data, segmentation, color, filters, io
from matplotlib import pyplot as plt
import sys
from itertools import product

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
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


def convert(path, compactness=30, segments=400, thresh=0.08):
    img = io.imread(path)
    edges = filters.sobel(color.rgb2gray(img))
    labels = segmentation.slic(img, compactness=compactness, n_segments=segments, start_label=1)
    g = graph.rag_boundary(labels, edges)

    # graph.show_rag(labels, g, img)
    # plt.title('Initial RAG')

    labels2 = graph.merge_hierarchical(labels, g, thresh=THRESH, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)

    # graph.show_rag(labels, g, img)
    # plt.title('RAG after hierarchical merging')

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    return out

def show(img):
    plt.figure()
    plt.imshow(img)
    plt.title('Final segmentation')
    plt.show()

def convert_grid(path):
    comp = (10, 20, 30, 40)
    seg = (400, 500, 600, 700, 800)
    thr = (0.08, 0.04, 0.01)

    fig, axs = plt.subplots(nrows=len(seg), ncols=len(comp)*len(thr), figsize=(12, 8),
                            subplot_kw={'xticks': [], 'yticks': []})

    for (compactness, segments, thresh), ax in zip(product(comp, seg, thr), axs.flat):
        img = convert(path, compactness, segments, thresh)
        ax.imshow(img)
        ax.set_title(f"C,S,T=({compactness}, {segments}, {thresh})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    PATH = sys.argv[1]
    COMPACTNESS = int(sys.argv[2]) # 30
    SEGMENTS = int(sys.argv[3]) # 400
    THRESH = float(sys.argv[4]) # 0.08

    #show(convert(PATH, COMPACTNESS, SEGMENTS, THRESH))

    convert_grid(PATH)
