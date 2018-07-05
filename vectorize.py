import numpy as np
np.set_printoptions(linewidth=np.nan, precision=3)
import cv2
import potrace
import matplotlib.pyplot as plt
from rdp import rdp
from face_data import load_faces
from thinning import guo_hall_thinning
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import networkx as nx
from tqdm import tqdm

def convert_to_3_stroke(im, eps=None):
    """
    params:
        im          image

    method:
        1)  dilate and erode the image to
            group line segments together
        2)  convert to bitmap
        3)  trace bitmap to SVG
        4)  convert SVG to 3-stroke format

    returns:
        strokes     3-stroke format
    """
    # black background, white sketch
    _, thresh = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY_INV)

    # dilate -> erode
    kernel = np.ones((3,3), np.uint8)
    im = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Get inverted boolean matrix:
    # potrace only works with boolean images
    data = im == 0

    # Create a bitmap from the array
    bmp = potrace.Bitmap(data)
    path = bmp.trace()

    # plt.imshow(data, cmap=plt.cm.gray)

    # get the xy coordinates for each curve
    lines = []
    for i, curve in enumerate(path):
        # for some reason, the first curve
        # is always weird
        if i == 0:
            continue

        line = curve.tesselate()

        # perform Ramer-Douglas-Peuker algorithm
        if eps:
            line = rdp(line, epsilon=eps)

        x, y = line.T
        plt.plot(x, y, c='red')

        lines.append(line)

    plt.show()

    # get the 3 stroke format
    strokes = lines_to_strokes(lines)

    return strokes

def get_opt_path(points):
    """
    params:
        points      unordered list of [x, y] coordinates

    method:
    - create a cyclic neighbor-graph
    - iterate over starting points to see where we can minimize distance

    returns:
        best_order  best order to read coordinates
    """
    # make a graph connecting points to nearest two nodes
    clf = NearestNeighbors(2).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)

    # order nodes by shortest distance, starting from each point
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]

    best_order = None
    best_length = 0
    best_cost = float('inf')

    # choose best path based on: min-distance and max-path-length
    for path in paths:
        ordered = points[path]
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()

        if cost < best_cost and len(ordered) >= best_length:
            best_cost = cost
            best_order = path
            best_length = len(ordered)

    return best_order

def get_window_3_stroke(im, j, i,
                        window_shape=(100,100),
                        scale_factor=10,
                        show=False):
    """
    a windowed function for the 3-stroke conversion

    params:
        lines           get_curves(im)
        im              image
        j, i            window coordinates
        window_shape    window dimensions (H, W)
        scale_factor    scale the strokes
        show            display the image and path

    method:
    - use Guo-Hall thinning to reduce to a skeleton
    - get all non-zero points in skeleton
    - use DBSCAN clustering to find clusters of neighbors
        - interpret each cluster as a "stroke"
    - use get_opt_path on each cluster to find best-fit line
    - simplify best-fitting lines with Ramer-Douglas-Peuker algorithm
    - convert lines to strokes

    returns:
        strokes         pen-stroke format
    """
    # preprocess window with Guo-Hall thinning
    window = im[j:j+window_shape[0], i:i+window_shape[1]]
    _, th = cv2.threshold(window,127,255,cv2.THRESH_BINARY_INV)
    window = guo_hall_thinning(th)

    points = np.argwhere(window.T != 0)
    # points = np.flip(points, 1)

    if len(points) == 0:
        return

    # segment graph into clusters using DBSCAN algorithm
    db = DBSCAN(eps=5)
    labels = db.fit_predict(points)

    lines = []
    if show:
        plt.imshow(window, 'gray')
    # for each cluster, get the optimal path
    for label in set(labels):
        cluster = points[labels==label]

        if len(cluster) < 3:
            continue

        path = get_opt_path(cluster)
        line = rdp(cluster[path], epsilon=1)

        # line = cluster[path]
        if show:
            x, y = line.T
            plt.plot(x, y)
        # add line to lines
        lines.append(line)

    if show:
        plt.show()

    strokes = lines_to_strokes(lines)

    # normalize strokes
    if len(strokes) > 0:
        strokes[:,0:2] /= scale_factor
        strokes[0] = [0, 0, 0]

    return strokes

if __name__ == '__main__':
    for face in load_faces(n=5):
        im = cv2.imread(face, 0)
        # strokes = convert_to_3_stroke(im)
        # lines = get_curves(im)
        lines = get_window_3_stroke(im, 0, 0)
        raise
        # draw_strokes(strokes)
