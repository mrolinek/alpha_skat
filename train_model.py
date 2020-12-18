from cluster import cluster_main

import pytorch_lightning as pl

from nn_utils.data_modules import PolicyDataModule, ValueDataModule
from nn_utils.trainers import PolicyModel, ValueModel


@cluster_main
def main(working_dir, num_epochs, model_params, data_params, mode):

    if mode == 'policy':
        data_module = PolicyDataModule(**data_params)
        model = PolicyModel(**model_params)
    elif mode == 'value':
        data_module = ValueDataModule(**data_params)
        model = ValueModel(**model_params)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20, default_root_dir=working_dir)
    trainer.fit(model, data_module)
    print(model.metrics)
    return model.metrics


if __name__ == "__main__":
    main()
