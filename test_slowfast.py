import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.loggers import TensorBoardLogger

import pytorchvideo.models as models
import torchvision.models.video as video

warnings.filterwarnings('ignore')
from dataset_slowfast import CustomDataModule
from model_slowfast import Videomodel
import torch.nn as nn
import torch
import timm
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set Model', add_help=False)
    parser.add_argument('--batch_size', default=1, type = int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--accelerator', default='cuda', type=str)
    parser.add_argument('--strategy', default=None, type=str)
    return parser

def create_model(pretrained=True):
    model_name = 'slowfast_r50'
    model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
    model.blocks[0].multipathway_blocks[0].conv= nn.Conv3d(4, 64, kernel_size=(1,7,7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    model.blocks[0].multipathway_blocks[1].conv= nn.Conv3d(4, 8, kernel_size=(5,7,7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
    model.blocks[6].proj = nn.Linear(2304, 1, bias=True)
    return model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Script', parents=[get_args_parser()])
    args = parser.parse_args()
    backbone = create_model(model_name=args.model_name)
    transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained = True)
    # logger = TensorBoardLogger('./tensorboard_log' )
    checkpoint_root = os.path.join('./checkpoint', args.model_name,'last.ckpt')
    # checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_root, save_top_k=1, monitor='val_loss', mode = 'min', save_last=True)
    model = Videomodel.load_from_checkpoint(backbone = backbone, transformer = transformer, device=args.accelerator, checkpoint_path=checkpoint_root)
    data_module = CustomDataModule(batch_size = args.batch_size, num_workers = args.num_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.device, max_epochs=args.epochs)
    trainer.test(model, data_module)