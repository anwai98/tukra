import os
import warnings

import torch


def declare_paths(nnunet_path: str):
    """To let the system known of the path variables where the respective folders exist (important for all components)

    Args:
        nnunet_path: The path where the expected files will be used by nnUNet for training and inference.
    """
    warnings.warn(
        "Make sure you have created the directories mentioned in this functions (relative to the root directory)"
    )

    os.environ["nnUNet_raw"] = os.path.join(nnunet_path, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(nnunet_path, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(nnunet_path, "nnUNet_results")


def preprocess_data(dataset_id: str, planner: str = "nnUNetPlannerResEncL"):
    """Function to plan and preprocess the inputs for nnUNet and prepare the training structure.

    Args:
        dataset_id: The dataset id created for the respective nnUNet-style data folder format.
        planner: The planner to design the experiments for training nnUNet.
    """
    # Let's check the preprocessing first
    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -pl {planner} --verify_dataset_integrity"
    os.system(cmd)


def train_nnunetv2(
    dataset_name: str,
    dataset_id: str,
    dim: str,
    plans: str = "nnUNetResEncUNetLPlans",
    fold: int = 0,
    have_prepared_splits: bool = True,
    compile_model: bool = False,
):
    """Function to train nnUNet in the expected structure.

    Args:
        dataset_name: The entire name of the dataset in nnUNet-style (say, `Dataset999_XYZ`)
        dataset_id: The dataset id created for the respective nnUNet-style data folder format.
        dim: The nnUNet configuration (2d / 3d_fullres) to train.
        plans: The plan designed for the experiments for training nnUNet.
        fold: The fold of choice for the 5-fold cross-validation based nnUNet training.
        have_prepared_splits: Whether the train-val splits and the respective split file for training are prepared.
    """
    if have_prepared_splits:
        # It's expected that you create folds by yourself in your desired train-val structure.
        _split_file_exists = os.path.exists(
            os.path.join(os.environ.get("nnUNet_preprocessed"), dataset_name, "splits_final.json")
        )
        assert _split_file_exists, "The experiment expects you to create the splits yourself."

    # Train your own nnUNet
    gpus = torch.cuda.device_count()
    _compile = "T" if compile_model else "f"
    cmd = f"nnUNet_compile={_compile} nnUNet_n_proc_DA=8 nnUNetv2_train {dataset_id} {dim} {fold} -num_gpus {gpus} --c "
    cmd += f"-p {plans}"
    os.system(cmd)


def predict_nnunetv2(
    dataset_name: str,
    dataset_id: str,
    dim: str,
    plans: str = "nnUNetResEncUNetLPlans",
    fold: int = 0,
    save_probabilities: bool = False,
):
    """Function to predict using trained nnUNet in the expected structure.

    Args:
        dataset_name: The entire name of the dataset in nnUNet-style (say, `Dataset999_XYZ`)
        dataset_id: The dataset id created for the respective nnUNet-style data folder format.
        dim: The nnUNet configuration (2d / 3d_fullres) to train.
        plans: The plan designed for the experiments for training nnUNet.
        fold: The fold for training nnUNet.
        save_probabilities: Whether to store the predicted raw probabilities.
    """
    input_dir, output_dir = _get_inference_paths(dataset_name, fold)
    assert os.path.exists(input_dir), "The input folder does not exists. Please preprocess the input images first."

    cmd = f"nnUNetv2_predict -i {input_dir} -o {output_dir} -d {dataset_id} -c {dim} -f {fold} -p {plans} "
    if save_probabilities:
        cmd += "--save_probabilities"

    os.system(cmd)


def _get_inference_paths(dataset_name, fold):
    output_dir = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "predictionTs", f"fold_{fold}")
    input_dir = os.path.join(os.environ.get("nnUNet_raw"), dataset_name, "imagesTs")
    return input_dir, output_dir
