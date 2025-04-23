import os
import cv2
import numpy as np
import joblib
from src.dataloader import load_mall_annotations, generate_head_bboxes
from src.footlocalizer import FootLocalizer
from matplotlib import pyplot as plt

def main():
    frame_files = sorted([os.path.join("mall_dataset/frames", f) for f in os.listdir("mall_dataset/frames")])
    frames = [cv2.imread(f) for f in frame_files]
    background = FootLocalizer.build_background(frames)
    localizer = FootLocalizer(background)
    annotations = load_mall_annotations("mall_dataset/mall_gt.mat")

    f1 = frames[0]
    mask = localizer._compute_binary_mask(f1)
    bboxes = generate_head_bboxes(annotations.iloc[0]['annotations'])
    feet = []
    for bbox in bboxes:
        loc = localizer.get_mask_localize_init(mask, bbox)
        if loc is not None:
            feet.append(loc)
    feet = np.array(feet)
    plt.imshow(mask)
    print(len(feet))
    plt.scatter(feet[:,0], feet[:,1])
    plt.show()

if __name__ == "__main__":
    main()