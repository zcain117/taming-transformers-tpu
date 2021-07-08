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

class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*256*256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3*256*256)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        myzeros = torch.zeros([256, 8192]).to(x)
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

batch_size=6
img_size=256
train_loader = xu.SampleGenerator(
    data=(torch.zeros(batch_size, 3, img_size , img_size ),
    torch.zeros(batch_size, dtype=torch.int64)),
    sample_count=1200000 // batch_size // xm.xrt_world_size())
val_loader = xu.SampleGenerator(
    data=(torch.zeros(batch_size, 3, img_size , img_size ),
    torch.zeros(batch_size, dtype=torch.int64)),
    sample_count=50000 // batch_size // xm.xrt_world_size())

# init model
autoencoder = LitAutoEncoder()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# trainer = pl.Trainer(gpus=8) (if you have GPUs)
trainer = pl.Trainer(tpu_cores=8)
trainer.fit(autoencoder, train_loader)
