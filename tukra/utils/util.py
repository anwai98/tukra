import os
from typing import Union


def _all_imageio_formats():
    import imageio
    all_formats = []
    for fmt in imageio.formats:
        all_formats.extend(imageio.formats[f"{fmt.name}"].extensions)

    return all_formats


def read_image(
    input_path: Union[os.PathLike, str],
    return_original_manifest: bool = False
):
    """Function to read popular biomedical imaging formats.
    Returns numpy arrays for each supported formats.
    """
    assert os.path.exists(input_path)

    extension = input_path.split(os.extsep, 1)[-1]
    extension = f".{extension}"

    if extension == ".nii" or extension == ".nii.gz":
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

    elif extension == ".nrrd":
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
