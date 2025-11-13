"""Results for OrgaSegment by running CellPose models:

1. CellPoseSAM:
    a. default: 0.383
    b. finetuned:

2. CellPose3:
    a. default:
    b. finetuned:

3. CellPose2:
    a: default:
    b. finetuned:
"""


import os
from tqdm import tqdm

import numpy as np
import imageio.v3 as imageio

from tukra.training.cellpose import run_cellposesam_finetuning
from tukra.inference.get_cellpose import segment_using_cellpose, segment_using_custom_cellpose

from elf.evaluation import mean_segmentation_accuracy


def get_organoid_data_paths(name, split):
    from torch_em.data.datasets.light_microscopy import orgasegment

    base_dir = "/mnt/lustre-grete/usr/u16934/data"

    if name == "orgasegment":
        image_paths, label_paths = orgasegment.get_orgasegment_paths(
            path=os.path.join(base_dir, name), split=split, download=True,
        )

    return image_paths, label_paths


def train_cellpose():
    # Get the image and corresponding labels' filepaths.
    train_image_paths, train_label_paths = get_organoid_data_paths(name="orgasegment", split="train")
    val_image_paths, val_label_paths = get_organoid_data_paths(name="orgasegment", split="val")

    # Train CellPoseSAM model.
    checkpoint_path = run_cellposesam_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_label_paths,
        val_image_files=val_image_paths,
        val_label_files=val_label_paths,
        save_root="./cellpose_finetuning",
        checkpoint_name="finetune_cpsam_orgasegment",
        n_epochs=10,
    )

    print(f"The model has been stored at '{checkpoint_path}'.")

    return checkpoint_path


def evaluate_cellpose(custom=None):
    # Get the image and corresponding labels.
    image_paths, label_paths = get_organoid_data_paths(name="orgasegment", split="eval")

    scores = []
    for curr_image_path, curr_label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Evaluate CellPose",
    ):

        # Laod the image.
        image = imageio.imread(curr_image_path)
        labels = imageio.imread(curr_label_path)

        if image.ndim != 2:
            image = image[:, :, 0]  # One channel is all we need!

        assert image.ndim == 2, "For CP, I am making assumptions so I just hard-code this expectation for now."

        if custom:  # custom trained model.
            masks = segment_using_custom_cellpose(image=image, checkpoint_path=custom)
        else:  # out-of-the-box validation.
            masks = segment_using_cellpose(image=image, model_choice="cpsam")

        # Let's do a simple evaluation.
        curr_msa = mean_segmentation_accuracy(masks, labels)
        scores.append(curr_msa)

    # Let's find the mean over all images.
    final_msa = np.mean(scores)
    print("The mean segmentation accuracy is:", final_msa)


def main():
    train = True
    if train:
        checkpoint_path = train_cellpose()
    else:
        checkpoint_path = None

    evaluate_cellpose(custom=checkpoint_path)


if __name__ == "__main__":
    main()
