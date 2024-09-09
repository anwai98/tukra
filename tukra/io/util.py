import os
from typing import Union

import numpy as np


def _all_imageio_formats():
    import imageio
    all_formats = []
    for fmt in imageio.formats:
        all_formats.extend(imageio.formats[f"{fmt.name}"].extensions)

    return all_formats


def read_image(
    input_path: Union[os.PathLike, str],
    extension: Union[str],
    return_original_manifest: bool = False
):
    """Function to read most popular biomedical imaging formats.

    Current supported formats:
        - nifti format (.nii, .nii.gz)
        - dicom format (.dcm)
        - metaimage header format (.mha)
        - nearly raw raster data format (.nrrd, .seg.nrrd)
        - all imageio-supported formats

    Args:
        input_path: The path to the input data.
        extension: The extension of the input data
        return_original_manifest: Returns the original data manifest.

    Returns the numpy array for each supported formats.
    """
    assert os.path.exists(input_path), input_path

    if extension[0] != ".":
        extension = f".{extension}"

    if extension in [".nii", ".nii.gz"]:
        import nibabel as nib
        inputs = nib.load(input_path)
        input_array = inputs.get_fdata()

    elif extension == ".mha":
        import SimpleITK as sitk
        inputs = sitk.ReadImage(input_path)
        input_array = sitk.GetArrayFromImage(inputs)

    elif extension == ".dcm":
        import pydicom as dicom
        inputs = dicom.dcmread(input_path)
        input_array = inputs.pixel_array

    elif extension in [".nrrd", ".seg.nrrd"]:
        import nrrd
        input_array, header = nrrd.read(input_path)
        inputs = input_array

    elif extension in _all_imageio_formats():
        import imageio.v3 as imageio
        inputs = imageio.imread(input_path)
        input_array = inputs

    else:
        raise ValueError(f"'{extension}' is not a supported extension to read images in 'tukra'.")

    if return_original_manifest:
        return input_array, inputs
    else:
        return input_array


def write_image(
    image: np.ndarray,
    dst_path: Union[os.PathLike, str],
    desired_fmt: str = ".nii.gz",
    **kwargs
):
    """Function to write arrays to most popular biomedical imaging formats.

    Current supported formats:
        - nifti format (.nii, .nii.gz)
        - all imageio-supported formats

    Args:
        input_path: The path to the input data.
        extension: The extension of the input data
        return_original_manifest: Returns the original data manifest.

    Returns the numpy array for each supported formats.
    """
    if desired_fmt in [".nii", ".nii.gz"]:
        import nibabel as nib
        image_nifti = nib.Nifti2Image(image, np.eye(4))
        nib.save(image_nifti, dst_path)

    elif desired_fmt in _all_imageio_formats():
        import imageio.v3 as imageio
        imageio.imwrite(dst_path, image, **kwargs)

    else:
        raise ValueError(f"'{desired_fmt}' is not a supported format to write images in 'tukra'.")
