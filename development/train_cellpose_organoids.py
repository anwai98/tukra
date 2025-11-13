import imageio.v3 as imageio

from tukra.inference.get_cellpose import segment_using_cellpose


def train_cellpose():
    ...


def evaluate_cellpose(custom=None):

    # Get the image and corresponding labels.
    image_paths, label_paths = ...

    for curr_image_path in tqdm(image_paths):

        if custom:  # custom trained model.
            ...
        else:  # out-of-the-box validation.
            masks = segment_using_cellpose(
                image=image
                model_choice="cpsam",
            )
        



def main():
    # train_cellpose()
    evaluate_cellpose()


if __name__ == "__main__":
    main()
