from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import random_split

from dataset import ChangeDetDataset, ChangeDetDatasetReduced
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
from model import ChangeDetNet, PatchChangeDet, PatchChangeDet128, PatchChangeDet128XL
from torchinfo import summary

torch.set_float32_matmul_precision("high")

checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename="PatchChangeDet128XL_v1-{epoch:02d}-{val_loss:.2f}",
    )
lr_callback = LearningRateMonitor(logging_interval="epoch")

dataset = ChangeDetDataset(r"D:\Work\tum\ChangeDet\data\pairs", reduce_to=128)

train_size = int(0.75 * len(dataset))
val_size = int((len(dataset) - train_size))
print("Train size: ", train_size)
print("Val size: ", val_size)

base_train_set, base_val_set = random_split(dataset, (train_size, val_size))
train_set = ChangeDetDatasetReduced(base_train_set, validation=False)
val_set = ChangeDetDatasetReduced(base_val_set, validation=True)

run_name = "PatchChangeDet128_v4"
run_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{run_name}'
print("STARTING RUN: ", run_name)
logger = TensorBoardLogger(save_dir="logs", name="", version=run_name)

model = PatchChangeDet128(train_set, val_set, batch_size=1)
summary(model, input_size=[(1, 1, 128, 128), (1, 1, 128, 128)])


trainer = Trainer(logger=logger, log_every_n_steps=50, callbacks=[checkpoint_callback, lr_callback], accelerator="gpu",detect_anomaly=False)
trainer.fit(model)

