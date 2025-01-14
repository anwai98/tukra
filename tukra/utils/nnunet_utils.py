import os
import shutil
from tqdm import tqdm
from typing import Union, Literal, Callable, List, Optional, Dict, Tuple

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
    ensure_unique: bool = False,
    keys: Tuple[str, str] = None,
) -> List[str]:
    """Functionality to ensure conversion of assorted filepaths to the input images and respective labels
    to convert them in common formats (eg. tif and nifti formats) for nnUNet training.

    Args:
        image_paths: List of filepaths for the image data.
        gt_paths: List of filepaths for the label data.
        split: The choice of data split.
        dataset_name: The name of dataset in nnUNet-style.
        file_suffix: The filepath extension for images.
        transfer_mode: The mode of transferring inputs in source path(s) to target path(s).
        preprocess_inputs: A callable to convert the input images.
        preprocess_labels: A callable to convert the input labels.
        ensure_unique: Whether to force all images to have unique id values.
        keys: Whether the inputs are stored in container formats under specific hierarchy names.

    Returns:
        List of filenames for the particular split.
    """
    if keys is not None and len(keys) != 2:
        raise ValueError("The 'keys' argument expects a tuple of keynames for both image and corresponding labels.")

    # The idea is to move all images into specific desired directory,
    # Write their image ids into a 'split.json' file,
    # which nnUNet will read to define the custom (fixed) validation split.
    image_dir = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "imagesTs" if split == "test" else "imagesTr")
    gt_dir = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "labelsTs" if split == "test" else "labelsTr")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    assert len(image_paths) == len(gt_paths)
    ids = []
    counter = 0
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc="Preprocessing inputs"):
        image_id = os.path.basename(image_path)
        image_id = image_id.split(".")[0]

        if file_suffix[0] != ".":
            file_suffix = "." + file_suffix

        _split = (split + f"_{counter}") if ensure_unique else split
        counter += 1

        fname = f"{image_id}_{_split}"
        target_image_path = os.path.join(image_dir, f"{fname}_0000{file_suffix}")
        target_gt_path = os.path.join(gt_dir, f"{fname}{file_suffix}")
        ids.append(fname)

        if os.path.exists(target_image_path) and os.path.exists(target_gt_path):
            continue

        if transfer_mode == "copy":
            shutil.copy(src=image_path, dst=target_image_path)
            shutil.copy(src=gt_path, dst=target_gt_path)

        elif transfer_mode == "store":
            image = read_image(image_path, key=keys if keys is None else keys[0])
            if preprocess_inputs is not None:
                image = preprocess_inputs(image)

            gt = read_image(gt_path, key=keys if keys is None else keys[1])
            if preprocess_labels is not None:
                gt = preprocess_labels(gt)

            write_image(image=image, dst_path=target_image_path)
            write_image(image=gt, dst_path=target_gt_path)

        else:
            raise ValueError(f"'{transfer_mode}' is not a supported transfer mode.")

    if len(ids) != len(gt_paths):
        raise AssertionError(
            f"Num. of input labels don't match the expected num. of converted labels. '{len(ids)}; {len(gt_paths)}'"
        )

    return ids


def create_json_files(
    dataset_name: str,
    file_suffix: str,
    dataset_json_template: Dict,
    train_ids: List[Union[os.PathLike, str]],
    val_ids: Optional[List[Union[os.PathLike, str]]] = None,
):
    """Functionality to create:
    1. `dataset.json` file, which moderates the input metadata (eg. count of data, dataset description, label ids, etc.)
    2. (OPTIONAL) `splits_final.json` file, which ensures consistent train-val splits (subjected to `val_ids`).
        By default, performs cross-validation on the entire train-set.

    Args:
        dataset_name: The name of dataset in nnUNet-style.
        file_suffix: The filepath extension for images.
        dataset_json_template: The json template for creating `dataset.json`, necessary for nnUNet training.
        train_ids: The list of filenames for training.
        val_ids: The list of filenames for validation.
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
