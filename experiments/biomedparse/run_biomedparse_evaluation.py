import os
from glob import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torch_em.data.datasets.histopathology.pannuke import get_pannuke_paths

from tukra.inference import get_biomedparse
from tukra.io import read_image, write_image
from tukra.evaluation import evaluate_predictions


def _preprocess_pannuke_dataset():
    res_dir = "results"
    if os.path.exists(res_dir):
        image_paths = glob(os.path.join(res_dir, "image_*"))
        gt_paths = glob(os.path.join(res_dir, "gt_*"))
        return image_paths, gt_paths

    os.makedirs(res_dir, exist_ok=True)

    input_path = get_pannuke_paths(
        path="/mnt/vast-nhr/projects/cidas/cca/data/pannuke", folds=["fold_3"], download=True
    )[0]

    with h5py.File(input_path, "r") as f:
        images = f["images"][:]
        labels = f["labels/semantic"][:]

    images = images.transpose(1, 2, 3, 0)  # make channels last

    for i, (image, label) in enumerate(zip(images, labels)):
        print(image.shape, label.shape)
        write_image(os.path.join(res_dir, f"image_{i}.png"), image.astype("uint8"))
        write_image(os.path.join(res_dir, f"gt_{i}.png"), label.astype("uint8"))

        # HACK: We test it for 10 images for now.
        if i > 10:
            break

    image_paths = glob(os.path.join(res_dir, "image_*"))
    gt_paths = glob(os.path.join(res_dir, "gt_*"))

    return image_paths, gt_paths


def evaluate_segmentation(segmentation, gt):
    # Evaluate predictions per semantic class.
    for i, seg in enumerate(segmentation, start=1):  # TODO: refactor this for semantic evaluation.
        dice = evaluate_predictions(prediction=seg, ground_truth=gt, metrics=["dice_score"])
        print(dice)


def visualize_results(image, gt, segmentation, prompts):
    def _generate_colors(n):
        cmap = plt.get_cmap('tab10')
        colors = [tuple(int(255 * val) for val in cmap(i)[:3]) for i in range(n)]
        return colors

    colors = _generate_colors(len(prompts))

    legend_patches = [
        mpatches.Patch(color=np.array(color) / 255, label=prompt) for color, prompt in zip(colors, prompts)
    ]

    fig, axes = plt.subplots(1, 2 + len(prompts), figsize=(20, 10))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(gt)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    axes[1].legend(handles=legend_patches, loc='upper right', fontsize='small')

    print(segmentation.shape)

    axc = 2
    for i, prompt in enumerate(prompts):
        axes[axc].imshow(segmentation[i])
        axes[axc].set_title(f"Predictions: '{prompt}'")
        axes[axc].axis('off')
        axes[axc].legend(handles=legend_patches, loc='upper right', fontsize='small')
        axc += 1

    plt.savefig("./test.png")
    plt.close()


def main():
    model = get_biomedparse.get_biomedparse_model(
        checkpoint_path="/mnt/vast-nhr/projects/cidas/cca/models/biomedparse/biomedparse_v1.pt"
    )

    # Preprocess PanNuke images
    image_paths, gt_paths = _preprocess_pannuke_dataset()

    # Choice of prompts
    prompts = ["nuclei"]

    for image_path, gt_path in zip(image_paths, gt_paths):
        image = read_image(image_path)
        gt = read_image(gt_path)

        segmentation = get_biomedparse.run_biomedparse_inference(image, prompts, model)
        evaluate_segmentation(segmentation, gt)
        visualize_results(image, gt, segmentation, prompts)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
