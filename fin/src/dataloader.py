import scipy.io as sio
import pandas as pd
import numpy as np

def load_mall_annotations(gt_path):
    """Load ground-truth head annotations from .mat file."""
    gt = sio.loadmat(gt_path)
    framedata = pd.DataFrame({
        'hc': gt['count'].flatten().astype(int),
        'annotations': [i[0][0][0] for i in gt['frame'][0]],
    }).reset_index().rename({'index': 'fn'}, axis=1)
    return framedata

def generate_head_bboxes(annotations):
    """Convert head annotations to bounding boxes."""
    _, bboxes = _h2bbox(annotations)
    return bboxes

def _h2bbox(points):
    """Helper: Convert head points to bounding boxes."""
    plotboxes, bboxes = [], []
    for point in points:
        bw = bh = 5
        ul = [point[0]-bw, point[1]-bw]
        br = [point[0]+bw, point[1]+bh]
        bboxes.append([ul, br])
    return np.array(plotboxes), np.array(bboxes)