from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json

import pytorch_lightning as pl
from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


model_name = "regnetz_c16"
optimizer_name = "ADAM"
learning_rate = 0.000012
batch_size = 64


def train(model, datamodule):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="output")
    early_stop_callback = EarlyStopping(monitor="val/acc", min_delta=0.00, patience=3, verbose=False, mode="max")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[early_stop_callback],
        logger=[tb_logger]
    )
    hyperparameters = dict(model_name=model_name, optimizer_name=optimizer_name, learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule)
    
    return trainer

def save_scripted_model(model):
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script,"model.scripted.pt")
    
    
def save_model(model):
    torch.save(model.state_dict(), "model.pt")


def save_last_ckpt(trainer):
    trainer.save_checkpoint("last.ckpt")


