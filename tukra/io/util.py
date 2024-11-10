import os
from pathlib import Path
from typing import Union, Optional

import numpy as np


def _all_imageio_formats():
    import imageio
    all_formats = set()

    for fmt in imageio.formats:
        all_formats.update(fmt.extensions)

    return all_formats


def read_image(
    input_path: Union[os.PathLike, str],
    extension: Optional[str] = None,
    key: Optional[str] = None,
    return_original_manifest: bool = False
) -> np.ndarray:
    """Function to read most popular biomedical imaging formats.

    Current supported formats:
        - nifti format (nii, nii.gz)
        - dicom format (dcm)
        - metaimage header format (mha)
        - nearly raw raster data format (nrrd, seg.nrrd)
        - all imageio-supported formats
        - all elf-supported formats (hdf5, zarr, n5, knossos)

    Args:
        input_path: The path to the input data.
        extension: The extension of the input data.
        key: The hierarchy name for container-based data structures.
        return_original_manifest: Returns the original data manifest.

    Returns:
        The numpy array.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    if extension is not None and key is not None:
        raise ValueError("You must choose either an extension or the key.")

    if key is not None:
        from elf.io import open_file
        inputs = open_file(path=input_path, ext=extension)[key]
        input_array = inputs[:]

    else:
        if extension is None:
            ipath = Path(input_path.rstrip("/"))
            suffixes = ipath.suffixes
        else:
            if extension[0] != ".":
                extension = f".{extension}"

            suffixes = [extension]

        if ".nii" in suffixes or ".nii.gz" in suffixes:
            import nibabel as nib
            inputs = nib.load(input_path)
            input_array = inputs.get_fdata()

        elif ".nrrd" in suffixes or ".seg.nrrd" in suffixes:
            import nrrd
            input_array, header = nrrd.read(input_path)
            inputs = input_array

        elif suffixes[-1] == ".mha":
            import SimpleITK as sitk
            inputs = sitk.ReadImage(input_path)
            input_array = sitk.GetArrayFromImage(inputs)

        elif suffixes[-1] == ".dcm":
            import pydicom as dicom
            inputs = dicom.dcmread(input_path)
            input_array = inputs.pixel_array

        elif len(set(suffixes) - _all_imageio_formats()) == 0:
            import imageio.v3 as imageio
            inputs = imageio.imread(input_path)
            input_array = inputs

        else:
            raise ValueError(f"'{suffixes}' is not a supported extension to read images in 'tukra'.")

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
