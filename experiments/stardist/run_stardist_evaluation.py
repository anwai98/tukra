import os
from glob import glob
from natsort import natsorted

import pandas as pd

from tukra.io import read_image
from tukra.evaluation import evaluate_predictions
from tukra.inference import segment_using_stardist


def evaluate_dsb(data_dir, chosen_metrics):
    image_paths = natsorted(glob(os.path.join(data_dir, "test", "images", "*.tif")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "test", "masks", "*.tif")))

    all_results = []
    for image_path, gt_path in zip(image_paths, gt_paths):
        image = read_image(image_path)
        gt = read_image(gt_path)

        segmentation = segment_using_stardist(image=image, model_name="2D_versatile_fluo")

        scores = evaluate_predictions(segmentation, gt, metrics=chosen_metrics)
        all_results.append(pd.DataFrame.from_dict([scores]))

    results = pd.concat(all_results)

    mean_results = {}
    for metric in chosen_metrics:
        mean_results[metric] = results[metric].mean()

    print(mean_results)


def main():
    chosen_metrics = ["msa", "sa50", "sa75"]

    data_dir = "/mnt/vast-nhr/projects/cidas/cca/data/dsb"
    evaluate_dsb(data_dir, chosen_metrics)


if __name__ == "__main__":
    main()
