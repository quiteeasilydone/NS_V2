import torch
import torch.nn as nn

def create_model():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model.blocks[0].conv = nn.Conv3d(4, 64, kernel_size=(1,7,7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    model.blocks[5].proj = nn.Linear(2048, 1, bias=True)

    return model