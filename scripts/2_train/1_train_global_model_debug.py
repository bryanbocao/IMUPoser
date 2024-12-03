# %%
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('../../src')

# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import torch

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser

# set the random seed
seed_everything(42, workers=True)

parser = get_parser()
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment

# %%
# config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoser",
#                 project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
#                 r6d=True, loss_type="mse", use_joint_loss=True, device="0")

config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoser",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mpjve", use_joint_loss=True, device="0")

# %%
# instantiate model and data
model = get_model(config)

# model.load_from_checkpoint('/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10252024-190150/epoch=epoch=0-val_loss=validation_step_loss=0.02620.ckpt')
# model.load_from_checkpoint('/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10112024-113938/epoch=epoch=62-val_loss=validation_step_loss=0.01087.ckpt')
# model.load_from_checkpoint('/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10112024-113938/epoch=epoch=62-val_loss=validation_step_loss=0.01087.ckpt')

# IMUPoserGlobalModel_global-10112024-113938/epoch=epoch=62-val_loss=validation_step_loss=0.01087.ckpt

checkpoint_path = '/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10252024-190150/epoch=epoch=0-val_loss=validation_step_loss=0.02620.ckpt'
checkpoint_path = '/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/checkpoints/IMUPoserGlobalModel_global-10252024-193756/epoch=epoch=67-val_loss=validation_step_loss=0.00992.ckpt'

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
print('\nmodel: ', model)
print('\ndir(model): ', dir(model))

datamodule = get_datamodule(config, test_only=True)
print('\ndatamodule: ', datamodule)
print('\ndir(datamodule): ',  dir(datamodule))
checkpoint_path = config.checkpoint_path 

# %%
# wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

# early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
#                                         min_delta=0.00001, patience=5)
# checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
#                                       save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
#                                       filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

# trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=1000, accelerator="gpu", devices=[0],
#                      callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)

trainer = pl.Trainer(fast_dev_run=fast_dev_run, max_epochs=1000, accelerator="gpu", devices=[0], deterministic=True)

# print('\ndir(trainer): ', dir(trainer))

# %%
# trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)

# %%
# with open(checkpoint_path / "best_model.txt", "w") as f:
#     f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
