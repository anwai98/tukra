import os
from typing import List, Optional, Union, Literal

from cellpose import train, core, io, models


def run_cellpose2_finetuning(
    train_image_files: List[Union[os.PathLike, str]],
    train_label_files: List[Union[os.PathLike, str]],
    val_image_files: List[Union[os.PathLike, str]],
    val_label_files: List[Union[os.PathLike, str]],
    model_name: Optional[str] = None,
    save_root: Optional[Union[os.PathLike, str]] = None,
    initial_model: str = "cyto2",
    n_epochs: int = 100,
    channels_to_use_for_training: Literal["Grayscale", "Blue", "Green", "Red"] = "Grayscale",
    second_training_channel: Literal["Blue", "Green", "Red", "None"] = "None",
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
    batch_size: int = 8,
    optimizer_choice: Literal["AdamW", "SGD"] = "AdamW",
    **kwargs
):
    """Functionality for finetuning (or training) CellPose models.

    This script is inspired from: https://github.com/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb.

    NOTE: The current support is to finetune (or train) CellPose models according to "CellPose 2.0":
    - Pachitariu et al. - https://doi.org/10.1038/s41592-022-01663-4

    Please cite it if you use this functionality in your research.

    Args:
        train_image_files (List[os.PathLike, str]): List of paths of the training image files.
        train_label_files (List[os.PathLike, str]): List of paths of the training image files.
        val_image_files (List[os.PathLike, str]): List of paths of the training image files.
        val_label_files (List[os.PathLike, str]): List of paths of the training image files.
        save_root (str, os.PathLike): Where to save the trained model.
        model_name (str, None): The name of model with which it will be saved.
        initial_model (str): The pretrained model to initialize CellPose with for finetuning (or, train from scratch).
        n_epochs (int): The total number of epochs for training.
        channels_to_use_for_training (str): The first channel to be used for training.
        second_training_channel (str): The second channel to be used for training.
        learning_rate (float): The learning rate fro training.
        weight_decay (float): The weight decay for the optimizer.
        batch_size: The number of patches to batch together per iteration.

    Returns:
        diam_labels: The diameter of objects in labels in the training set.
    """
    # Here we match the channel to number
    channels_to_use_for_training = channels_to_use_for_training.title()
    if channels_to_use_for_training == "Grayscale":
        chan = 0
    elif channels_to_use_for_training == "Blue":
        chan = 3
    elif channels_to_use_for_training == "Green":
        chan = 2
    elif channels_to_use_for_training == "Red":
        chan = 1
    else:
        raise ValueError(f"'{chan}' is not a valid channel to use for training.")

    second_training_channel = second_training_channel.title()
    if second_training_channel == "Blue":
        chan2 = 3
    elif second_training_channel == "Green":
        chan2 = 2
    elif second_training_channel == "Red":
        chan2 = 1
    elif second_training_channel == "None":
        chan2 = 0
    else:
        raise ValueError(f"'{chan2}' is not a valid second training channel.")

    # In case we would like to train CellPose from scratch.
    if initial_model == "scratch":
        initial_model = "None"

    # Check whether GPU is activated or not.
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    # start logger (to see training across epochs)
    io.logger_setup()

    # Let's define the CellPose model (without size model)
    model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

    new_model_path = train.train_seg(
        net=model.net,
        train_files=train_image_files,
        train_labels_files=train_label_files,
        test_files=val_image_files,
        test_labels_files=val_label_files,
        channels=[chan, chan2],
        save_path=save_root,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        SGD=(optimizer_choice == "SGD"),
        batch_size=batch_size,
        model_name=model_name,
        momentum=momentum,
        **kwargs
    )

    # Diameter of labels in the training images (useful for evaluation)
    diam_labels = model.net.diam_labels.item()

    return new_model_path, diam_labels
