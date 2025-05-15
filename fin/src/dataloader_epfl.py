import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment



# TODO: RUN FOOTLOCALIZER ON EPFL, USE DATALOADER TO VALIDATE. REFER TO TESTING.PY FOR GETTING FOOTLOCALIZER TO WORK
# USE VALIDATE_SINGLEFRAMESINGLECAM TO FIND DIFFERENCES WHEN VALIDATING



def parse_camview(path):
    fd = defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            xmin, ymin, xmax, ymax = map(int, parts[1:5])
            frame = int(parts[5])
            midpoint_x = (xmin+xmax)/2
            fd[frame].append([midpoint_x, ymax])
    
    max_frame = max(fd.keys())
    all_points = []

    for frame in range(max_frame+1):
        if frame in fd:
            all_points.append(np.array(fd[frame]))
        else:
            all_points.append(np.empty((0,2)))

    return all_points

def validate_singleframesinglecam(set1, set2):
    diff = set1[:,np.newaxis,:] - set2[np.newaxis,:,:]
    dist_mat = np.linalg.norm(diff, axis=2)

    row_ind, row_col = linear_sum_assignment(dist_mat)
    total = dist_mat[row_ind, row_col].sum()

    return total