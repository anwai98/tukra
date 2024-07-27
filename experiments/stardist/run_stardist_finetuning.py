from stardist import gputools_available
from stardist.models import Config2D, StarDist2D


def run_stardist_finetuning(
    train_image_paths,
    train_gt_paths,
    val_image_paths,
    val_gt_paths,
):
    n_rays = 32

    use_gpu = False and gputools_available()

    grid = (2, 2)

    n_channel = 1

    configuration = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
    )
    print(configuration)

    model = StarDist2D(configuration, name='stardist', basedir='models')
    model.train(
        X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter
    )

    # Threshold optimization
    model.optimize_thresholds(X_val, Y_val)
