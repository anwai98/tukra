import os
import shutil
from tqdm import tqdm
from pathlib import Path
from typing import Union, Literal, Callable, List, Optional, Dict

import json

from tukra.io import read_image, write_image


def convert_dataset_for_nnunet_training(
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    split: Literal['train', 'val', 'test'],
    dataset_name: str,
    file_suffix: str,
    transfer_mode: Literal["copy", "store"],
    preprocess_inputs: Optional[Callable] = None,
    preprocess_labels: Optional[Callable] = None,
):
    """
    """
    # The idea is to move all images into specific desired directory,
    # Write their image ids into a 'split.json' file,
    # which nnUNet will read to define the custom (fixed) validation split.
    image_dir = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "imagesTs" if split == "test" else "imagesTr")
    gt_dir = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "labelsTs" if split == "test" else "labelsTr")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    assert len(image_paths) == len(gt_paths)
    ids = []
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths), desc="Preprocessing inputs"
    ):
        image_id = os.path.basename(image_path)
        image_id = image_id.split(".")[0]

        if file_suffix[0] != ".":
            file_suffix = "." + file_suffix

        target_image_path = os.path.join(image_dir, f"{image_id}_{split}_0000{file_suffix}")
        target_gt_path = os.path.join(gt_dir, f"{image_id}_{split}{file_suffix}")

        if transfer_mode == "copy":
            shutil.copy(src=image_path, dst=target_image_path)
            shutil.copy(src=gt_path, dst=target_gt_path)

        elif transfer_mode == "store":
            image = read_image(image_path)
            if preprocess_inputs is not None:
                image = preprocess_inputs(image)

            gt = read_image(gt_path)
            if preprocess_labels is not None:
                gt = preprocess_labels(gt)

            write_image(image=image, dst_path=target_image_path)
            write_image(image=gt, dst_path=target_gt_path)

        else:
            raise ValueError(f"'{transfer_mode}' is not a supported transfer mode.")

        ids.append(Path(target_gt_path).stem)

    return ids


def create_json_files(
    dataset_name: str,
    file_suffix: str,
    dataset_json_template: Dict,
    train_ids: List[os.PathLike, str],
    val_ids: Optional[List[os.PathLike, str]] = None,
):
    """
    """
    # First, let's create the 'datasets.json' file based on the available inputs.
    if file_suffix[0] != ".":
        file_suffix = "." + file_suffix

    json_file = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "dataset.json")

    with open(json_file, "w") as f:
        json.dump(dataset_json_template, f, indent=4)

    # Let's store the split files.
    preprocessed_dir = os.path.join(os.environ.get("nnUNet_preprocessed"), dataset_name)
    os.makedirs(preprocessed_dir, exist_ok=True)

    if val_ids is not None:
        # Create custom splits for all folds - to fit with the expectation.
        all_split_inputs = [{'train': train_ids, 'val': val_ids} for _ in range(5)]
        json_file = os.path.join(preprocessed_dir, "splits_final.json")
        with open(json_file, "w") as f:
            json.dump(all_split_inputs, f, indent=4)
