"""Results (Mean Segmentation Accuracy) for OrgaSegment by running CellPose models:

NOTE: All results are for OrgaSegment, unless mentioned specifically.

1. CellPoseSAM:
    a. default (evaluated on OrgaSegment-test): 0.383
    b. finetuned (trained on OrgaSegment-train and evaluated on OrgaSegment-eval, 100 epochs): 0.525
    c. finetuned (trained on Internal-train and evaluated on OrgaSegment-test, 100 epochs):

    d. default (evaluated on Internal-test): 0.254
    e. finetuned (trained on Internal-train and evaluated on Internal-test, 100 epochs):
    f. finetuned (trained on OrgaSegment-train and evaluated on Internal-eval, 100 epochs): 0.223

2. CellPose3:
    a. default: 0.284
    b. w. denoiser: 0.287
    c. w. deblur: 0.256
    d. w. upsampler: 0.289
    e. finetuned: 0.447

3. CellPose2:
    a: default: 0.294
    b. finetuned: 0.474

NOTE: How to make install CellPose without a massive mess?
1. CellPoseSAM - the first thing is to create a new environment and install `micro-sam`:
    - `micromamba create -n cpsam -c conda-forge python=3.11 micro_sam`
    - `pip install cellpose`
2. CellPose 3 / CellPose2 - same as above, but stick to latest CellPosev3 release for this:
    - `micromamba create -n cp3 -c conda-forge python=3.11 micro_sam`
    - `pip install "cellpose<4"`
"""

import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio

from tukra.training.cellpose import run_cellposesam_finetuning, run_cellpose2_finetuning
from tukra.inference.get_cellpose import segment_using_cellpose, segment_using_custom_cellpose

from elf.evaluation import mean_segmentation_accuracy


def get_organoid_data_paths(name, split):
    from torch_em.data.datasets.light_microscopy import orgasegment

    if name == "orgasegment":
        base_dir = "/mnt/lustre-grete/usr/u16934/data"
        if not os.path.exists(base_dir):  # Switch to the other project username.
            base_dir = "/mnt/lustre-grete/usr/u12090/data"

        image_paths, label_paths = orgasegment.get_orgasegment_paths(
            path=os.path.join(base_dir, name), split=split, download=True,
        )
    elif name == "internal":
        base_dir = "/mnt/lustre-grete/usr/u12090/data/orga_v3_umg_sartorius/"  # NOTE: Copied data to cache flows.

        split = "test" if split == "eval" else split  # HACK: Simple hard-coding to work with both data.
        assert split in ["train", "val", "test"], "The provided split is not valid."

        image_paths = natsorted(glob(os.path.join(base_dir, split, "images", "*")))
        image_paths = [p for p in image_paths if "_flows.tif" not in p]  # Filter out flow paths
        label_paths = natsorted(glob(os.path.join(base_dir, split, "masks", "*")))

        assert image_paths and len(image_paths) == len(label_paths)

        # HACK: Just train or evaluate on first 10 images for a simple design.
        if split == "test":
            image_paths, label_paths = image_paths[:10], label_paths[:10]

    return image_paths, label_paths


def train_cellposesam(data_name="orgasegment"):
    # Get the image and corresponding labels' filepaths.
    train_image_paths, train_label_paths = get_organoid_data_paths(name=data_name, split="train")
    val_image_paths, val_label_paths = get_organoid_data_paths(name=data_name, split="val")

    # Train CellPoseSAM model.
    checkpoint_path = run_cellposesam_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_label_paths,
        val_image_files=val_image_paths,
        val_label_files=val_label_paths,
        save_root="./cellpose_finetuning",
        checkpoint_name=f"finetune_cpsam_{data_name}",
        n_epochs=100,
        min_train_masks=2,
    )

    print(f"The model has been stored at '{checkpoint_path}'.")

    return checkpoint_path


def train_cellpose3(data_name="orgasegment"):
    # Get the image and corresponding labels' filepaths.
    train_image_paths, train_label_paths = get_organoid_data_paths(name=data_name, split="train")
    val_image_paths, val_label_paths = get_organoid_data_paths(name=data_name, split="val")

    # Train CellPose3 model (same backbone as CP2)
    checkpoint_path, _ = run_cellpose2_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_label_paths,
        val_image_files=val_image_paths,
        val_label_files=val_label_paths,
        save_root="./cellpose_finetuning/",
        checkpoint_name=f"finetune_cyto3_{data_name}",
        initial_model="cyto3",
        n_epochs=10,
    )
    checkpoint_path = str(checkpoint_path[0])

    print(f"The model has been stored at '{checkpoint_path}'.")

    return checkpoint_path


def train_cellpose2(data_name="orgasegment"):
    # Get the image and corresponding labels' filepaths.
    train_image_paths, train_label_paths = get_organoid_data_paths(name=data_name, split="train")
    val_image_paths, val_label_paths = get_organoid_data_paths(name=data_name, split="val")

    # Train CellPose2 model.
    checkpoint_path, _ = run_cellpose2_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_label_paths,
        val_image_files=val_image_paths,
        val_label_files=val_label_paths,
        save_root="./cellpose_finetuning/",
        checkpoint_name=f"finetune_cyto2_{data_name}",
        initial_model="cyto2",
        n_epochs=10,
    )
    checkpoint_path = str(checkpoint_path[0])

    print(f"The model has been stored at '{checkpoint_path}'.")

    return checkpoint_path


def evaluate_cellpose(model_choice, data_name="orgasegment", custom=None):
    # Get the image and corresponding labels.
    image_paths, label_paths = get_organoid_data_paths(name=data_name, split="eval")

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
            masks = segment_using_custom_cellpose(image=image, diameter=None, channels=None, checkpoint_path=custom)
        else:  # out-of-the-box validation.
            masks = segment_using_cellpose(image=image, model_choice=model_choice)

        # Let's do a simple evaluation.
        curr_msa = mean_segmentation_accuracy(masks, labels)
        scores.append(curr_msa)

    # Let's find the mean over all images.
    final_msa = np.mean(scores)
    print("The mean segmentation accuracy is:", final_msa)


def main():
    train = False
    if train:
        checkpoint_path = train_cellposesam(data_name="internal")
        # checkpoint_path = train_cellpose3(data_name="internal")
        # checkpoint_path = train_cellpose2(data_name="internal")
    else:
        checkpoint_path = None

    # HACK:
    checkpoint_path = "./cellpose_finetuning/models/finetune_cpsam_internal"

    # NOTE: Change the `model_choice` to 'cyto2' / 'cyto3' / 'cpsam'.
    evaluate_cellpose(model_choice="cpsam", data_name="internal", custom=checkpoint_path)


if __name__ == "__main__":
    main()
