# Results

import os
from glob import glob
from natsort import natsorted

import pandas as pd

from tukra.io import read_image
from tukra.evaluation import evaluate_predictions
from tukra.inference import segment_using_instanseg


def run_instanseg_for_dsb(data_dir, chosen_metrics):
    image_paths = natsorted(glob(os.path.join(data_dir, "images", "*.tif")))
    gt_paths = natsorted(glob(os.path.join(data_dir, "masks", "*.tif")))

    all_results = []
    for image_path, gt_path in zip(image_paths, gt_paths):
        image = read_image(image_path)
        segmentation = segment_using_instanseg(
            image=image,
            model_type="fluorescence_nuclei_and_cells",
            target="nuclei",
        )

        gt = read_image(gt_path)
        scores = evaluate_predictions(segmentation, gt, metrics=chosen_metrics)
        all_results.append(pd.DataFrame.from_dict([scores]))

    results = pd.concat(all_results)

    mean_results = {}
    for metric in chosen_metrics:
        mean_results[metric] = results[metric].mean()

    print(mean_results)


def main():
    chosen_metrics = ["mSA", "sa50", "sa75"]

    data_dir = "/media/anwai/ANWAI/data/dsb/test"
    run_instanseg_for_dsb(data_dir, chosen_metrics)


if __name__ == "__main__":
    main()
