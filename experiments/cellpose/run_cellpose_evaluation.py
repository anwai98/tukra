# Results on LIVECell test-set (msa):
# 'cyto': 0.1448
# 'cyto2': 0.1559
# 'cyto3': 0.4289
# 'cyto3' + 'denoise_cyto3': 0.4031
# 'livecell': 0.4407
# 'livecell_cp3': 0.4415


from tqdm import tqdm

import pandas as pd

from torch_em.data.datasets.light_microscopy import livecell, orgasegment

from tukra.io import read_image
from tukra.evaluation import evaluate_predictions
from tukra.inference import segment_using_cellpose


def run_cellpose_for_livecell(data_dir, chosen_metrics):
    image_paths, gt_paths = livecell.get_livecell_data(path=data_dir, split="test", download=False)

    all_results = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path)
        gt = read_image(gt_path)

        segmentation = segment_using_cellpose(image=image, model_choice="cyto3")

        scores = evaluate_predictions(segmentation, gt, metrics=chosen_metrics)
        all_results.append(pd.DataFrame.from_dict([scores]))

    results = pd.concat(all_results)

    mean_results = {}
    for metric in chosen_metrics:
        mean_results[metric] = results[metric].mean()

    print(mean_results)


def run_cellpose_for_orgasegment(data_dir, chosen_metrics):
    image_paths, gt_paths = orgasegment._get_data_paths(path=data_dir, split="eval", download=False)

    all_results = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = read_image(image_path)
        gt = read_image(gt_path)

        segmentation = segment_using_cellpose(image=image, model_choice="cyto2")

        scores = evaluate_predictions(segmentation, gt, metrics=chosen_metrics)
        all_results.append(pd.DataFrame.from_dict([scores]))

    results = pd.concat(all_results)

    mean_results = {}
    for metric in chosen_metrics:
        mean_results[metric] = results[metric].mean()

    print(mean_results)


def main():
    chosen_metrics = ["mSA", "sa50", "sa75"]

    # data_dir = "/scratch/projects/nim00007/sam/data/livecell/"
    # run_cellpose_for_livecell(data_dir=data_dir, chosen_metrics=chosen_metrics)

    data_dir = "/scratch/share/cidas/cca/data/orgasegment_cp"
    run_cellpose_for_orgasegment(data_dir=data_dir, chosen_metrics=chosen_metrics)


if __name__ == "__main__":
    main()
