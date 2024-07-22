from cellpose import train
from cellpose import core, io, models


def run_cellpose_2_finetuning(
    train_dir,
    test_dir,
    save_root=None,
    model_name="",
    initial_model="cyto2",
    n_epochs=100,
    channels_to_use_for_training="Grayscale",
    second_training_channel=None,
    learning_rate=0.1,
    weight_decay=0.0001,
    use_default_advanced_parameters=True,
):
    # Here we match the channel to number
    if channels_to_use_for_training == "Grayscale":
        chan = 0
    elif channels_to_use_for_training == "Blue":
        chan = 3
    elif channels_to_use_for_training == "Green":
        chan = 2
    elif channels_to_use_for_training == "Red":
        chan = 1

    if second_training_channel == "Blue":
        chan2 = 3
    elif second_training_channel == "Green":
        chan2 = 2
    elif second_training_channel == "Red":
        chan2 = 1
    elif second_training_channel == "None":
        chan2 = 0

    if initial_model == "scratch":
        initial_model = "None"

    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    # start logger (to see training across epochs)
    logger = io.logger_setup()

    # DEFINE CELLPOSE MODEL (without size model)
    model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

    # set channels
    channels = [chan, chan2]

    # get files
    output = io.load_train_test_data(train_dir, test_dir, mask_filter='_seg.npy')
    train_data, train_labels, _, test_data, test_labels, _ = output

    new_model_path = train.train_seg(
        net=model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        channels=channels,
        save_path=save_root,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        SGD=True,
        nimg_per_epoch=8,
        model_name=model_name,
    )

    # diameter of labels in training images
    diam_labels = model.net.diam_labels.item()
