import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pytorch_lightning.loggers import TensorBoardLogger

import pytorchvideo.models as models
import torchvision.models.video as video

warnings.filterwarnings('ignore')
from dataset import CustomDataModule
from model import Videomodel
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
    parser.add_argument('--model_name',default='slow_r50', type=str)
    return parser

def create_model(model_name, pretrained=True):

    if model_name == 'slow_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        model.blocks[0].conv = nn.Conv3d(4, 64, kernel_size=(1,7,7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        model.blocks[5].proj = nn.Linear(2048, 1, bias=True)
        return model
    
    elif model_name == 'slowfast_r50':
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        model.blocks[0].conv = nn.Conv3d(4, 64, kernel_size=(1,7,7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        model.blocks[6].proj = nn.Linear(2304, 1, bias=True)
        return model           
    
    elif model_name == 'x3d_s':
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        model.blocks[0].conv_t = nn.Conv3d(4, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        model.blocks[5].proj = nn.Linear(2048, 1, bias=True)
        return model         
                    
    elif model_name == 'csn':
        model = models.create_csn()
        model.blocks[0].conv = nn.Conv3d(4, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        model.blocks[5].proj = nn.Linear(2048, 1, bias=True)
        return model 
        
    elif model_name == 'r2plus1d':
        model = video.r2plus1d_18(pretrained = True)
        model.stem[0] = nn.Conv3d(4, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        model.fc = nn.Linear(512, 1, bias=True)
        return model     

    elif model_name == 'mc3_18':
        model = video.mc3_18(pretrained = True)
        model.stem[0] = nn.Conv3d(4, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        model.fc = nn.Linear(512, 1, bias=True)
        return model     
            
    elif model_name == 's3d':
        model = video.s3d(pretrained = True)
        model.features[0][0][0] = nn.Conv3d(4, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        model.classifier[1] = nn.Conv3d(1024, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        return model      
            
    elif model_name == 'mvit_v1_b':
        model = video.mvit_v1_b(pretrained = True)
        model.conv_proj = nn.Conv3d(4, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
        model.head[1] = nn.Linear(768, 1, bias=True)
        return model
            
    elif model_name == 'mvit_v2_s':
        model = video.mvit_v2_s(pretrained = True)
        model.conv_proj = nn.Conv3d(4, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
        model.head[1] = nn.Linear(768, 1, bias=True)
        return model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Script', parents=[get_args_parser()])
    args = parser.parse_args()
    backbone = create_model(model_name=args.model_name)
    transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained = True)
    # logger = TensorBoardLogger('./tensorboard_log' )
    checkpoint_root = './checkpoint'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_root, save_top_k=1, monitor='val_loss', mode = 'min', save_last=True)

    model = Videomodel(backbone = backbone, transformer = transformer, device=args.accelerator)
    data_module = CustomDataModule(batch_size = args.batch_size, num_workers = args.num_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.device, max_epochs=args.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)