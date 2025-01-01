import os
from pathlib import Path
from typing import Union, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from torch_em.data.datasets.light_microscopy.neurips_cell_seg import to_rgb  # TODO: Make a generic function out of this.  # noqa

from biomedparse import configs
from biomedparse.modeling import build_model
from biomedparse.modeling.BaseModel import BaseModel
from biomedparse.utilities.constants import BIOMED_CLASSES
from biomedparse.modeling.language.loss import vl_similarity
from biomedparse.utilities.distributed import init_distributed
from biomedparse.utilities.arguments import load_opt_from_config_files

from tukra.io import read_image


@torch.no_grad()
def run_biomedparse_inference(
    input_path: Union[os.PathLike, str, np.ndarray], text_prompts: Optional[List[str]], model: torch.nn.Module,
) -> np.ndarray:
    """Scripts to run inference for BioMedParse.

    Args:
        input_path: Filepath to the input image.
        text_prompts: The choice of text prompts.
        model: The model for inference.

    Returns:
        The segmentation masks.
    """
    if not isinstance(input_path, np.ndarray):
        image = read_image(input_path)
    else:
        image = input_path

    # Ensure RGB images.
    image = to_rgb(image)

    # Dimensions of original image
    height = image.shape[-2]
    width = image.shape[-1]

    image = torch.from_numpy(image).cuda()  # convert inputs to tensors.

    # Transform image to (1024, 1024)
    transform = transforms.Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC)
    image = transform(image)

    # Initialize the task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = True
    model.model.task_switch['audio'] = False
    model.model.task_switch['grounding'] = True

    # The line below enables batched inference.
    batch_inputs = [
        {"image": image, 'text': text_prompts, "height": height, "width": width}
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

    # NOTE: Below is commented out as it is not used for inference atm.
    # Reference: https://github.com/microsoft/BiomedParse/issues/46
    # pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    # Resize the inputs to the original input shape
    pred_mask_prob = F.interpolate(
        pred_masks_pos[None,], (height, width), mode='bilinear'
    )[0, :, :height, :width].sigmoid().cpu().numpy()

    # Binarise the predictions.
    pred_mask = (1 * (pred_mask_prob > 0.5)).astype(np.uint8)
    print(f"Shape of predicted masks: {pred_mask.shape}")

    return pred_mask


def get_biomedparse_model(checkpoint_path: Union[os.PathLike, str]) -> torch.nn.Module:
    """Get the BioMedParse model.

    Args:
        checkpoint_path: Filepath where the model checkpoint is stored.

    Returns:
        The segmentation model.
    """
    config_file = str(Path(configs.__file__).parent / "biomedparse_inference.yaml")
    opt = load_opt_from_config_files([config_file])
    opt = init_distributed(opt)

    model = BaseModel(opt, build_model(opt)).from_pretrained(checkpoint_path).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    return model
