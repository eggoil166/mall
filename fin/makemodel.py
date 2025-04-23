import os
import cv2
import numpy as np
import joblib
from src.dataloader import load_mall_annotations, generate_head_bboxes
from src.footlocalizer import FootLocalizer

def densestats():
    # Feed a few sparse frames here to initialize the gp model
    frame_files = sorted([os.path.join("mall_dataset/frames", f) for f in os.listdir("mall_dataset/frames")])
    frames = [cv2.imread(f) for f in frame_files]
    background = FootLocalizer.build_background(frames)
    localizer = FootLocalizer(background)
    annotations = load_mall_annotations("mall_dataset/mall_gt.mat")
    # assume that the thing is std gauss
    densities = []
    for i, frame in enumerate(frames):
        num_people = len(annotations.iloc[i])
        mask = localizer._compute_binary_mask(frame)
        density = localizer._estimate_density(mask, num_people)
        densities.append(density)
    densities = np.array(densities)
    print("mean, std: ", np.mean(densities), np.std(densities))

def main():
    frame_files = sorted([os.path.join("mall_dataset/frames", f) for f in os.listdir("mall_dataset/frames")])
    frames = [cv2.imread(f) for f in frame_files]
    background = FootLocalizer.build_background(frames)
    localizer = FootLocalizer(background)
    annotations = load_mall_annotations("mall_dataset/mall_gt.mat")

    for f in range(len(frames)):
        num_people = len(annotations.iloc[f]['annotations'])
        mask = localizer._compute_binary_mask(frames[f])
        density = localizer._estimate_density(mask, num_people)
        if density < localizer.gp_threshold:
            bboxes = generate_head_bboxes(annotations.iloc[0]['annotations'], f)
            for bbox in bboxes:
                loc = localizer.get_mask_localize_init(mask, bbox)
                if loc is not None:
                    localizer.add_training_pair(((bbox[0,0]+bbox[1,0])/2, (bbox[0,1]+bbox[1,1])/2), loc[1])
    
    print(localizer.head_foot_pairs)
    localizer.fit_gp_model()
    print("saving")
    localizer.save_gp_model("models/base.pkl")

if __name__ == "__main__":
    main()