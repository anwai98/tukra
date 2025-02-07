import os
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import nrrd
except ImportError:
    nrrd = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import pydicom as dicom
except ImportError:
    dicom = None

try:
    import slideio
except ImportError:
    slideio = None


ITEMPLATE = "conda install -c conda-forge"


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
    return_original_manifest: bool = False,
    scale: Optional[Tuple[int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Function to read most popular biomedical imaging formats.

    Current supported formats:
        - nifti format (.nii, nii.gz)
        - dicom format (.dcm)
        - metaimage header format (.mha)
        - nearly raw raster data format (.nrrd, seg.nrrd)
        - whole-slide image data formats (.svs)
        - all imageio-supported formats (eg. png, tif, jpg, etc.)
        - all elf-supported formats (hdf5, zarr, n5, knossos)

    Args:
        input_path: The path to the input data.
        extension: The extension of the input data.
        key: The hierarchy name for container-based data structures.
        return_original_manifest: Returns the original data manifest.
        scale: Relevant for WSIs, to get the image for a desired scale.
        image_size: Relevant for WSIs, to get a ROI crop for a desired shape.

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
            assert nib is not None, f"Please install 'nibabel': '{ITEMPLATE} nibabel'"
            inputs = nib.load(input_path)
            input_array = inputs.get_fdata()

        elif ".nrrd" in suffixes or ".seg.nrrd" in suffixes:
            assert nrrd is not None, f"Please install 'nrrd': '{ITEMPLATE} pynrrd'."
            input_array, header = nrrd.read(input_path)
            inputs = input_array

        elif suffixes[-1] == ".mha":
            assert sitk is not None, f"Please install 'SimpleITK': '{ITEMPLATE} simpleitk'."
            inputs = sitk.ReadImage(input_path)
            input_array = sitk.GetArrayFromImage(inputs)

        elif suffixes[-1] == ".dcm":
            assert dicom is not None, f"Please install 'pydicom': '{ITEMPLATE} pydicom'."
            inputs = dicom.dcmread(input_path)
            input_array = inputs.pixel_array

        elif suffixes[-1] in [".svs", ".afi", ".zvi", ".czi", ".ndpi", ".scn", ".vsi"]:
            assert slideio is not None, "Please install 'slideio': 'pip install slideio'."
            slide = slideio.open_slide(input_path)  # Fetches the slide object.

            # Let's check with expected scale.
            if scale is None:
                scale = (0, 0)  # Loads original resolution.
            else:
                if not isinstance(scale, Tuple) and len(scale) != 2:
                    raise ValueError(
                        "The scale parameter is expected to be a tuple of height and width dimensions, "
                        "such that the new shape is (H', W')"
                    )

            # Let's check for the expected size of the desired ROI.
            # NOTE: Here, we expect all values for placing an ROI precisely: (x, y, W, H)
            if image_size is None:
                image_size = (0, 0, 0, 0)
            else:
                if not isinstance(image_size, Tuple):
                    raise ValueError(
                        "The image size parameter is expected to be a tuple of desired target ROI crop, "
                        "such that the new crop shape is for this ROI."
                    )

                # If the user provides shapes in the usual 2d axes format, eg. (1024, 1024),
                # we provide them a top-left corner crop.
                if len(image_size) == 2:
                    image_size = (0, 0, *image_size)

            assert len(scale) == 2
            assert len(image_size) == 4

            # NOTE: Each slide objects contains one or multiple scenes (WSI resolution-stuff),
            # which is coined as a continuous raster region (with the 2d image, other meta-data, etc)
            scene = slide.get_scene(0)
            input_array = scene.read_block(size=scale, rect=image_size)
            inputs = scene  # We return one particular scene depending on the requested resolution.

        elif len(set(suffixes) - _all_imageio_formats()) == 0 or suffixes[-1] in _all_imageio_formats():
            import imageio.v3 as imageio
            inputs = imageio.imread(input_path)
            input_array = inputs

        else:
            raise ValueError(f"'{suffixes}' is not a supported extension to read images in 'tukra'.")

    if return_original_manifest:
        return input_array, inputs
    else:
        return input_array


def write_image(dst_path: Union[os.PathLike, str], image: np.ndarray, **kwargs):
    """Function to write arrays to most popular biomedical imaging formats.

    Current supported formats:
        - nifti format (.nii, .nii.gz)
        - all imageio-supported formats

    Args:
        dst_path: The destination path where the array will be written.
        image: The input array to write to the destination path.
    """
    ipath = Path(dst_path.rstrip("/"))
    suffixes = ipath.suffixes

    if ".nii" in suffixes or ".nii.gz" in suffixes:
        assert nib is not None, f"Please install 'nibabel': '{ITEMPLATE} nibabel'."
        image_nifti = nib.Nifti2Image(image, np.eye(4))
        nib.save(image_nifti, dst_path)

    elif len(set(suffixes) - _all_imageio_formats()) == 0:
        import imageio.v3 as imageio
        imageio.imwrite(dst_path, image, **kwargs)

    else:
        raise ValueError(f"'{suffixes}' is not a supported format to write images in 'tukra'.")
