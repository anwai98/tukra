import os
from pathlib import Path
from typing import Union, List, Optional, Dict

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# TODO: Make a generic function out of this.
from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb

from biomedparse import configs
from biomedparse.modeling import build_model
from biomedparse.modeling.BaseModel import BaseModel
from biomedparse.modeling.language.loss import vl_similarity
from biomedparse.utilities.distributed import init_distributed
from biomedparse.utilities.arguments import load_opt_from_config_files
from biomedparse. inference_utils.inference import non_maxima_suppression
from biomedparse.utilities.constants import BIOMED_CLASSES, BIOMED_OBJECTS
from biomedparse.inference_utils.output_processing import check_mask_stats, combine_masks

from tukra.io import read_image


def _load_tensor_from_input_array(input_path):
    if not isinstance(input_path, np.ndarray):
        image_array = read_image(input_path)
    else:
        image_array = input_path

    # Ensure RGB images.
    image_array = to_rgb(image_array)

    # Dimensions of original image
    height = image_array.shape[-2]
    width = image_array.shape[-1]

    image = torch.from_numpy(image_array).cuda()  # convert inputs to tensors.

    # Transform image to (1024, 1024)
    transform = transforms.Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC)
    image = transform(image)

    return image, image_array.transpose(1, 2, 0), height, width


@torch.no_grad()
def _run_biomedparse_batched_inference(image, height, width, model, prompts, verbose=True):
    # The line below enables batched inference.
    batch_inputs = [
        {"image": image, "text": prompts, "height": height, "width": width}
    ]
    results, image_size, extra = model.model.evaluate_demo(batch_inputs)

    # All stuff to ensure that the best matching predictions are selected subjected to input prompts.
    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id, :, :]

    # NOTE: Below is commented out as it is not used for inference atm. This returns the meta class.
    # Reference: https://github.com/microsoft/BiomedParse/issues/46
    # pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    # Resize the inputs to the original input shape.
    pred_mask_prob = F.interpolate(
        pred_masks_pos[None,], (height, width), mode='bilinear'
    )[0, :, :height, :width].sigmoid().cpu().numpy()

    # Binarise the predictions.
    pred_mask = (1 * (pred_mask_prob > 0.5)).astype(np.uint8)
    if verbose:
        print(f"Shape of predicted masks: {pred_mask.shape}")

    return pred_mask


def run_biomedparse_prompt_based_inference(
    input_path: Union[os.PathLike, str, np.ndarray], text_prompts: Optional[List[str]], model: torch.nn.Module,
) -> np.ndarray:
    """Scripts to run inference for BioMedParse using input text prompts.

    Args:
        input_path: Filepath to the input image.
        text_prompts: The choice of text prompts.
        model: The model for inference.

    Returns:
        The segmentation masks.
    """
    image, _, height, width = _load_tensor_from_input_array(input_path)

    # Initialize the task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = True
    model.model.task_switch['audio'] = False
    model.model.task_switch['grounding'] = True

    # Run batched inference.
    pred_mask = _run_biomedparse_batched_inference(image, height, width, model, text_prompts)

    return pred_mask


def run_biomedparse_automatic_inference(
    input_path: Union[os.PathLike, str, np.ndarray],
    modality_type: str,
    model: torch.nn.Module,
    p_value_threshold: Optional[float] = None,
    batch_size: int = 1,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Scripts to run automatic inference for BioMedParse.

    Args:
        input_path: Filepath to the input image.
        image_type: The choice of imaging modality.
        model: The model for inference.
        p_value_threshold: The p-value used to pre-filter the masks for the imaging modality.
        batch_size: The batch size for running inference.
        verbosity: Whether to have verbosity for model outputs.

    Returns:
        A dictionary with semantic class and corresponding masks.
    """
    image, image_array, height, width = _load_tensor_from_input_array(input_path)

    if modality_type not in BIOMED_OBJECTS:
        raise ValueError(f"Currently support modality types: '{list(BIOMED_OBJECTS.keys())}'")

    image_targets = BIOMED_OBJECTS[modality_type]

    predicts, p_values = {}, {}
    for i in range(0, len(image_targets), batch_size):
        # The line below enables batched inference.
        batch_targets = image_targets[i: i+batch_size]

        # Run batched inference.
        pred_mask = _run_biomedparse_batched_inference(image, height, width, model, batch_targets, verbose)

        # Iterate through the predictions for a specific modality by prompting the model
        # for a set of masks and merge them in the following step.
        for j, target in enumerate(batch_targets):
            adj_p_value = check_mask_stats(
                img=image_array,
                mask=pred_mask[j] * 255,
                modality_type=modality_type,
                target=target,
                target_dist_dir=os.path.join(_get_biomedparse_cachedir(), "inference_utils"),
                config_dir=os.path.join(_get_biomedparse_cachedir(), "configs"),
            )
            if p_value_threshold and adj_p_value < p_value_threshold:
                if verbose:
                    print(f"Reject null hypothesis for {target} with p-value {adj_p_value}")
                continue

            predicts[target] = pred_mask[j]
            p_values[target] = adj_p_value

    predicts = non_maxima_suppression(predicts, p_values)
    masks = combine_masks(predicts)

    return {target: mask for target, mask in masks.items()}


def _get_biomedparse_cachedir():
    # Define the cache dir.
    cache_dir = os.path.join(Path.home(), ".cache", "biomedparse")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_biomedparse_model(checkpoint_path: Optional[Union[os.PathLike, str]] = None) -> torch.nn.Module:
    """Get the BioMedParse model.

    NOTE: With the source installation for BioMedParse, `huggingface-cli` is already installed.
    You should follow the steps below to add token to automatically download model checkpoints.
    - Visit the hugging face website to create a new token: https://huggingface.co/settings/tokens.
    - Run the CLI to login: `huggingface-cli login`.
    - The CLI asks you to provide your authentication token next. Copy paste it and press enter.
    - Now your token is cached for working with huggingface hub supported downloads.

    Args:
        checkpoint_path: Filepath where the model checkpoint is stored.

    Returns:
        The segmentation model.
    """
    # Build the model config.
    config_file = str(Path(configs.__file__).parent / "biomedparse_inference.yaml")
    opt = load_opt_from_config_files([config_file])
    opt = init_distributed(opt)

    # Get the cache directory.
    cache_dir = _get_biomedparse_cachedir()

    if checkpoint_path is None:  # Download the model from hugging face.
        checkpoint_path = 'hf_hub:microsoft/BiomedParse'

    # Load model from pretrained weights.
    model = BaseModel(opt=opt, module=build_model(opt))
    model = model.from_pretrained(
        pretrained=checkpoint_path,
        local_dir=os.path.join(cache_dir, "pretrained"),
        config_dir=os.path.join(cache_dir, "configs"),
    )
    model.eval().cuda()

    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    return model
