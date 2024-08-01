import os
from glob import glob
from natsort import natsorted

from tukra.training import run_stardist_training


def train_dsb():
    path = "/scratch/share/cidas/cca/data/dsb"

    train_image_paths = natsorted(glob(os.path.join(path, "train", "images", "*.tif")))
    train_gt_paths = natsorted(glob(os.path.join(path, "train", "masks", "*.tif")))
    val_image_paths = natsorted(glob(os.path.join(path, "test", "images", "*.tif")))
    val_gt_paths = natsorted(glob(os.path.join(path, "test", "masks", "*.tif")))

    run_stardist_training(
        train_image_paths=train_image_paths,
        train_label_paths=train_gt_paths,
        val_image_paths=val_image_paths,
        val_label_paths=val_gt_paths,
        use_gpu=True,
    )


train_dsb()
