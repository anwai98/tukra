import imageio.v3 as imageio

from tukra.datasets import fetch_livecell_paths
from tukra.models import segment_using_cellpose
from tukra.evaluation import evaluate_predictions


def run_cellpose_for_livecell(data_dir):
    image_paths, gt_paths = fetch_livecell_paths(data_dir)
    for image_path, gt_path in zip(image_paths, gt_paths):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        segmentation = segment_using_cellpose(
            image=image,
            model_choice="cyto3",
            restoration_choice="denoise_cyto3"
        )

        scores = evaluate_predictions(segmentation, gt, metrics=["msa", "sa50", "sa75"])


def main():
    data_dir = "/scratch/projects/nim00007/sam/data/livecell/"
    run_cellpose_for_livecell(data_dir)


if __name__ == "__main__":
    main()
