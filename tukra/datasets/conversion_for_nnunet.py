import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
from typing import Optional, List, Union, Callable, Literal, Dict

import json

from ..io import read_image, write_image


class ConvertInputsToNifti:
    """Converts the inputs to a desired file format.
    """
    def __init__(
        self,
        src_extension: str,
        dst_extension: str = ".nii.gz",
        make_channels_first: bool = False,
    ):
        self.src_extension = src_extension
        self.dst_extension = dst_extension
        self.make_channels_first = make_channels_first

    def __call__(
        self, input_path: Union[os.PathLike, str], target_path: Union[os.PathLike, str]
    ):
        input_array = read_image(input_path=input_path, extension=self.src_extension)

        if self.make_channels_first:
            input_array = input_array.transpose(2, 0, 1)

        write_image(image=input_array, dst_path=target_path, desired_fmt=self.dst_extension)


def convert_images_and_labels_for_nnunet(
    root_dir: Union[os.PathLike, str],
    dataset_name: str,
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    split_name: Literal["train", "val", "test"] = "train",
    convert_inputs_function: Optional[Callable] = None,
):
    """Function to convert the images and respective labels in a desired format.

    The idea of this function is to move all the images to the expected directory.
    In addition, write a `split.json` file which nnUNet will read to define the custom validation split, if desired.

    Args:
        root_dir: The path where the converted images and corresponding labels will be stored.
        dataset_name: The name of dataset in nnUNet-style.
        image_paths: The list of filepaths for the image data.
        gt_paths: The list of filepaths for the corresponding label data.
        split_name: The choice of data split.
        convert_inputs_function: A callable function to convert the images and labels.
    """
    # Let's set the expected directories.
    idirname = "imagesTs" if split_name == "test" else "imagesTr"
    gdirname = "labelsTs" if split_name == "test" else "labelsTr"

    image_dir = os.path.join(root_dir, "nnUNet_raw", dataset_name, idirname)
    gt_dir = os.path.join(root_dir, "nnUNet_raw", dataset_name, gdirname)

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    def _move_per_split(split, image_paths, gt_paths):
        _ids = []
        assert len(gt_paths) == len(image_paths)
        for image_path, gt_path in tqdm(
            zip(sorted(image_paths), sorted(gt_paths)),
            total=len(image_paths),
            desc=f"Converting the data for the '{split}' split"
        ):
            image_id = Path(image_path).stem
            image_id = image_id.split(".")[0]

            trg_image_path = os.path.join(image_dir, f"{image_id}_{split}_0000.nii.gz")
            trg_gt_path = os.path.join(gt_dir, f"{image_id}_{split}.nii.gz")

            if os.path.exists(trg_image_path) and os.path.exists(trg_gt_path):
                continue

            if convert_inputs_function is None:
                shutil.copy(src=image_path, dst=trg_image_path)
                shutil.copy(src=gt_path, dst=trg_gt_path)
            else:
                convert_inputs_function(input_path=image_path, target_path=trg_image_path)
                convert_inputs_function(input_path=gt_path, target_path=trg_gt_path)

            _ids.append(Path(trg_gt_path).stem)

        return _ids

    _move_per_split(split_name, image_paths, gt_paths)


class CreateJsonFileForCustomSplits:
    """Functionality to create JSON file for custom defined train-val splits.

    Args:
        root_dir: The path where the images and corresponding labels are stored.
        dataset_name: The name of dataset in nnUNet-style.
        channel_names: The dictionary with channel indices and names for input images.
        labels: The dictionary with label ids and names for input labels.
        description: A brief description for the training setup.
        extension: The choice of file extension.
        have_val: Whether the dataset has a predefined val set.
    """
    def __init__(
        self,
        root_dir: Union[os.PathLike, str],
        dataset_name: str,
        channel_names: Dict,
        labels: Dict,
        description: str,
        extension: str = ".nii.gz",
        have_val: bool = False,
    ):
        if extension[0] != ".":
            extension = f".{extension}"

        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.have_val = have_val
        self.extension = extension
        self.channel_names = channel_names
        self.labels = labels
        self.description = description

        self.dataset_json_fpath = os.path.join(root_dir, "nnUNet_raw", dataset_name, "dataset.json")
        self.splits_json_fpath = os.path.join(root_dir, "nnUNet_preprocessed", dataset_name, "splits_final.json")

        gt_dir = os.path.join(root_dir, "nnUNet_raw", dataset_name, "labelsTr")
        if self.have_val:
            self.imids = [
                [Path(_path).stem.split(".")[0] for _path in glob(os.path.join(gt_dir, f"*train{extension}"))],
                [Path(_path).stem.split(".")[0] for _path in glob(os.path.join(gt_dir, f"*val{extension}"))]
            ]
        else:
            self.imids = [Path(_path).stem.split(".")[0] for _path in glob(os.path.join(gt_dir, f"*{extension}"))]

    def __call__(self):
        if self.have_val:
            train_ids, val_ids = self.imids
            n_images = len(train_ids) + len(val_ids)
        else:
            n_images = len(self.imids)

        data = {
            "channel_names": self.channel_names,
            "labels": self.labels,
            "numTraining": n_images,
            "file_ending": self.extension,
            "name": self.dataset_name,
            "description": self.description,
        }

        # This is the "dataset.json" file. This guides nnUNet to fetch the input data and respective labels.
        with open(self.dataset_json_fpath, "w") as f:
            json.dump(data, f, indent=4)

        if self.have_val:
            # If we have specific train-val splits, let's create the split file to guide nnUNet to train as desired.
            all_split_inputs = [{'train': train_ids, 'val': val_ids} for _ in range(5)]
            with open(self.splits_json_fpath, "w") as f:
                json.dump(all_split_inputs, f, indent=4)

        print(f"'dataset.json' is saved at '{self.dataset_json_fpath}'.")
