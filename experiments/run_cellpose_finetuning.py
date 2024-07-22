from torch_em.data.datasets.light_microscopy import orgasegment

from tukra.training import run_cellpose2_finetuning


def main():
    train_image_paths, train_gt_paths = orgasegment._get_data_paths(
        path="/scratch/share/cidas/cca/data/orgasegment_cp", split="train", download=True,
    )

    val_image_paths, val_gt_paths = orgasegment._get_data_paths(
        path="/scratch/share/cidas/cca/data/orgasegment_cp", split="val", download=False,
    )

    checkpoint_path, _ = run_cellpose2_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_gt_paths,
        val_image_files=val_image_paths,
        val_label_files=val_gt_paths,
        save_root="/scratch/share/cidas/cca/test/cellpose_finetuning/",
        checkpoint_name="finetune_cyto2_orgasegment",
    )

    print(f"The model has been stored at '{checkpoint_path}'")


if __name__ == "__main__":
    main()
