# Results on test-set (msa):
# 'cyto': 0.1448
# 'cyto2': 0.1559
# 'cyto3': 0.4289
# 'cyto3' + 'denoise_cyto3': 0.4031
# 'livecell': 0.4407
# 'livecell_cp3': 0.4415


from tqdm import tqdm

import pandas as pd
import imageio.v3 as imageio

from tukra.datasets import fetch_livecell_paths
from tukra.models import segment_using_cellpose
from tukra.evaluation import evaluate_predictions


def run_cellpose_for_livecell(data_dir, chosen_metrics):
    image_paths, gt_paths = fetch_livecell_paths(data_dir)

    all_results = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        segmentation = segment_using_cellpose(
            image=image,
            model_choice="cyto3",
        )

        scores = evaluate_predictions(segmentation, gt, metrics=chosen_metrics)
        all_results.append(pd.DataFrame.from_dict([scores]))

    results = pd.concat(all_results)

    mean_results = {}
    for metric in chosen_metrics:
        mean_results[metric] = results[metric].mean()

    print(mean_results)


def main():
    data_dir = "/scratch/projects/nim00007/sam/data/livecell/"
    run_cellpose_for_livecell(data_dir, chosen_metrics=["mSA", "sa50", "sa75"])


if __name__ == "__main__":
    main()
