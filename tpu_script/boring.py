import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        #self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        #self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
       # self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    global_rank = int(os.environ["CLOUD_TPU_TASK_ID"])
    default_root_dir = '/home/taehoon.kim/temp/'
    model = BoringModel()
    trainer = Trainer(
        default_root_dir = os.getcwd(),
        tpu_cores=8,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model, train_dataloader=train_data, val_dataloaders=val_data)
#    if global_rank == 0:
 #       trainer.save_checkpoint(default_root_dir+'example.ckpt')
    trainer.test(model, test_dataloaders=test_data)


if __name__ == '__main__':
    run()
