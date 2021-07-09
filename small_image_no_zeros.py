import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm


BATCH_SIZE = 6
IMAGE_SIZE = 28
NUM_CHANNELS = 3


class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(NUM_CHANNELS*IMAGE_SIZE*IMAGE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CHANNELS*IMAGE_SIZE*IMAGE_SIZE)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        print("in training step...")
        x, y = batch
        # Does not seem to affect the segfault.
        #myzeros = torch.zeros([256, 8192], device=self.device)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

train_loader = xu.SampleGenerator(
    data=(torch.zeros(BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
    torch.zeros(BATCH_SIZE, dtype=torch.int64)),
    sample_count=1200000 // BATCH_SIZE // xm.xrt_world_size())
val_loader = xu.SampleGenerator(
    data=(torch.zeros(BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
    torch.zeros(BATCH_SIZE, dtype=torch.int64)),
    sample_count=50000 // BATCH_SIZE // xm.xrt_world_size())

autoencoder = LitAutoEncoder()
trainer = pl.Trainer(tpu_cores=8)
trainer.fit(autoencoder, train_loader)
