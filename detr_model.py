import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

class attention_detr(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).eval()
        self.batch_size = batch_size
    
    def attention_score(self, x_list, batch_size):
        enc_attn_weights = []

        hooks = [self.model.transformer.encoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1]))]
        
        for i in range(batch_size):
            self.model(x_list[i])

        for hook in hooks:
            hook.remove()

        first_weight = enc_attn_weights.pop(0)
        first_weight = first_weight.unsqueeze(0)
        for weights in enc_attn_weights:
            weights = weights.unsqueeze(0)
            first_weight = torch.cat((first_weight, weights))
        
        enc_attn_weights = first_weight

        return enc_attn_weights
    
    def forward(self, x):
        return self.attention_score(x, self.batch_size)