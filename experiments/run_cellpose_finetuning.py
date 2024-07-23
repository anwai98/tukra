# Results on OrgaSegment test set:
# "cyto2":
#     - {'mSA': 0.29339636853771456, 'sa50': 0.473194573783817, 'sa75': 0.3060913601942939}
# finetuned "cyto2": (finetuned for 10 epochs)
#     - {'mSA': 0.4665593471680653, 'sa50': 0.7034327837020484, 'sa75': 0.5100150917921741}


from tqdm import tqdm

import pandas as pd
import imageio.v3 as imageio

from torch_em.data.datasets.light_microscopy import orgasegment

from tukra.evaluation import evaluate_predictions
from tukra.training import run_cellpose2_finetuning
from tukra.inference import segment_using_custom_cellpose


ROOT = "/scratch/share/cidas/cca/data/orgasegment_cp"


def finetune_orgasegment():
    train_image_paths, train_gt_paths = orgasegment._get_data_paths(path=ROOT, split="train", download=True)
    val_image_paths, val_gt_paths = orgasegment._get_data_paths(path=ROOT, split="val", download=False)

    checkpoint_path, _ = run_cellpose2_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_gt_paths,
        val_image_files=val_image_paths,
        val_label_files=val_gt_paths,
        save_root="/scratch/share/cidas/cca/test/cellpose_finetuning/",
        checkpoint_name="finetune_cyto2_orgasegment",
        initial_model="cyto2",
        n_epochs=10,
    )

    print(f"The model has been stored at '{checkpoint_path}'")

    return checkpoint_path


def evaluate_finetuned_model(checkpoint_path):
    chosen_metrics = ["mSA", "sa50", "sa75"]
    image_paths, gt_paths = orgasegment._get_data_paths(path=ROOT, split="eval", download=False)

    all_results = []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        segmentation = segment_using_custom_cellpose(image=image, checkpoint_path=checkpoint_path)

        scores = evaluate_predictions(segmentation, gt, metrics=chosen_metrics)
        all_results.append(pd.DataFrame.from_dict([scores]))

    results = pd.concat(all_results)

    mean_results = {}
    for metric in chosen_metrics:
        mean_results[metric] = results[metric].mean()

    print(mean_results)


def main():
    # Finetune CellPose "cyto2" model for OrgaSegment dataset.
    checkpoint_path = finetune_orgasegment()

    # Let's compare the finetuned model vs. "cyto2" model on OrgaSegment test set.
    evaluate_finetuned_model(checkpoint_path)


if __name__ == "__main__":
    main()
