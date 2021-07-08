import argparse, os, sys, datetime, glob, importlib
import numpy as np
import random
from PIL import Image
import torch

# vision imports

from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from taming.models.vqgan import VQModel, GumbelVQ

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, fake_data=False):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fake_data = fake_data
        self.img_size = img_size

        self.transform = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(self.img_size),
                                    T.CenterCrop(self.img_size),
                                    T.ToTensor()
                                    ])
                                    
    def setup(self, stage=None):
        if not self.fake_data:
            self.train_dataset = ImageFolder(self.train_dir, self.transform)
            self.val_dataset = ImageFolder(self.val_dir, self.transform)
  

    def train_dataloader(self):
        if self.fake_data:
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm
            train_loader = xu.SampleGenerator(
                            data=(torch.zeros(self.batch_size, 3, self.img_size , self.img_size ),
                            torch.zeros(self.batch_size, dtype=torch.int64)),
                            sample_count=1200000 // self.batch_size // xm.xrt_world_size())
            return train_loader
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):
        if self.fake_data:
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm            
            val_loader = xu.SampleGenerator(
                            data=(torch.zeros(self.batch_size, 3, self.img_size , self.img_size ),
                            torch.zeros(self.batch_size, dtype=torch.int64)),
                            sample_count=50000 // self.batch_size // xm.xrt_world_size())            
            return val_loader
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if self.fake_data:
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm
            val_loader = xu.SampleGenerator(
                            data=(torch.zeros(self.batch_size, 3, self.img_size , self.img_size ),
                            torch.zeros(self.batch_size, dtype=torch.int64)),
                            sample_count=50000 // self.batch_size // xm.xrt_world_size())            
            return val_loader            
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


    parser = argparse.ArgumentParser(description='VQGAN(Taming Transformers) for Pytorch TPU')

    #path configuration
    parser.add_argument('--train_dir', type=str, default='/home/taehoon.kim/taming-transformers-tpu/data/train/',
                    help='path to train dataset')
    parser.add_argument('--val_dir', type=str, default='/home/taehoon.kim/taming-transformers-tpu/data/val/',
                    help='path to val dataset')                    
    parser.add_argument('--log_dir', type=str, default='/home/taehoon.kim/results/',
                    help='path to save logs')
    parser.add_argument('--ckpt_path', type=str,default='/home/taehoon.kim/results/checkpoints/last.ckpt',
                    help='path to previous checkpoint')
    parser.add_argument('--refresh_rate', type=int, default=1,
                    help='progress bar refresh rate')  

    #training configuration
    parser.add_argument('--fake_data', action='store_true', default=False,
                    help='using fake_data for debugging') 
    parser.add_argument('--use_tpus', action='store_true', default=False,
                    help='using tpu') 
    parser.add_argument('--is_pod', action='store_true', default=False,
                    help='using tpu as pod')                                                                
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume from checkpoint')                   
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')  
    parser.add_argument('--gpus', type=int, default=16,
                    help='number of gpus')                   
    parser.add_argument('--num_sanity_val_steps', type=int, default=0,
                    help='num_sanity_val_steps')                     
    parser.add_argument('--learning_rate', default=4.5e-6, type=float,
                    help='base learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                    help='dataconfig')  
    parser.add_argument('--epochs', type=int, default=30,
                    help='dataconfig')                                    
    parser.add_argument('--num_workers', type=int, default=0,
                    help='dataconfig')   
    parser.add_argument('--img_size', type=int, default=256,
                    help='dataconfig')

    parser.add_argument('--test', action='store_true', default=False,
                    help='test run')                     

    #model configuration
    parser.add_argument('--model', type=str, default='vqgan')
    parser.add_argument('--embed_dim', type=int, default=256,
                    help='number of embedding dimension')       
    parser.add_argument('--n_embed', type=int, default=1024,
                    help='codebook size')        
    parser.add_argument('--double_z', type=bool, default=False,
                    help='ddconfig')
    parser.add_argument('--z_channels', type=int, default=256,
                    help='ddconfig')
    parser.add_argument('--resolution', type=int, default=256,
                    help='ddconfig')
    parser.add_argument('--in_channels', type=int, default=3,
                    help='ddconfig')
    parser.add_argument('--out_ch', type=int, default=3,
                    help='ddconfig')    
    parser.add_argument('--ch', type=int, default=128,
                    help='ddconfig')  
    parser.add_argument('--ch_mult', type=list, default=[1,1,2,2,4],
                    help='ddconfig')  
    parser.add_argument('--num_res_blocks', type=int, default=2,
                    help='ddconfig')                     
    parser.add_argument('--attn_resolutions', type=list, default=[16],
                    help='ddconfig')  
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='ddconfig')  

    #loss configuration
    parser.add_argument('--disc_conditional', type=bool, default=False,
                    help='lossconfig')      
    parser.add_argument('--disc_in_channels', type=int, default=3,
                    help='lossconfig') 
    parser.add_argument('--disc_start', type=int, default=250001,
                    help='lossconfig') 
    parser.add_argument('--disc_weight', type=float, default=0.8,
                    help='lossconfig') 
    parser.add_argument('--codebook_weight', type=float, default=1.0,
                    help='lossconfig') 

    #misc configuration
 
    args = parser.parse_args()

    #random seed fix
    seed_everything(args.seed)   

    #data = ImageDataModule(args.train_dir, args.val_dir, args.batch_size, args.num_workers, args.img_size, args.fake_data)

    transform = T.Compose([
                                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                                    T.Resize(args.img_size),
                                    T.CenterCrop(args.img_size),
                                    T.ToTensor()
                                    ])
    if not args.fake_data:
        train_dataset = ImageFolder(args.train_dir, transform)
        val_dataset = ImageFolder(args.val_dir, transform)   
    if args.fake_data:
        import torch_xla.utils.utils as xu
        import torch_xla.core.xla_model as xm        
        train_loader = xu.SampleGenerator(
                        data=(torch.zeros(args.batch_size, 3, args.img_size , args.img_size ),
                        torch.zeros(args.batch_size, dtype=torch.int64)),
                        sample_count=1200000 // args.batch_size // xm.xrt_world_size())
        val_loader = xu.SampleGenerator(
                        data=(torch.zeros(args.batch_size, 3, args.img_size , args.img_size ),
                        torch.zeros(args.batch_size, dtype=torch.int64)),
                        sample_count=50000 // args.batch_size // xm.xrt_world_size())                           
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)      
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)  

    # model
    if args.model == 'vqgan':
        model = VQModel(args, args.batch_size, args.learning_rate)
    elif args.model == 'gvqgan':
        model = GumbelVQ(args, args.batch_size, args.learning_rate)        

    default_root_dir = args.log_dir

    if args.resume:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = None

    if args.use_tpus:
        tpus = 8
        gpus = None
    else:
        tpus = None
        gpus = args.gpus

    if args.use_tpus:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=16,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          resume_from_checkpoint = ckpt_path)
    else:
        trainer = Trainer(tpu_cores=tpus, gpus= gpus, default_root_dir=default_root_dir,
                          max_epochs=args.epochs, progress_bar_refresh_rate=args.refresh_rate,precision=16,
                          accelerator='ddp',
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          resume_from_checkpoint = ckpt_path)
    
    print("Setting batch size: {} learning rate: {:.2e}".format(model.hparams.batch_size, model.hparams.learning_rate))
    
    if not args.test:    
        print("\n\n\njust before fit...\n\n\n")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.test(model, dataloaders=val_loader)


