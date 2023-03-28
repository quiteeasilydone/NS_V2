import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning
import pytorchvideo.models as models
import torchvision.models.video as video

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

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, model_name, batch_size):
        super().__init__()
        self.backbone = self.create_model(model_name=model_name)
        self.attention_weights = attention_detr(batch_size)
        self.batch_size = batch_size
        self.train_ratio = 0.8
        
    def create_model(self, model_name, pretrained=True):

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

    def forward(self, x):
        attention_input = torch.einsum('bcfhw -> bfchw', x)
        attention_mask = self.attention_weights(attention_input)

        attention_mask = attention_mask.unsqueeze(1)
        input_x = torch.cat((x, attention_mask), dim = 1)
        output = self.backbone(input_x)

        return output
    
    def training_step(self, batch, batch_idx):
        video_data, label = batch
        pred = self.forward(video_data)
        loss_function = nn.MSELoss()
        loss = loss_function(pred, label)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        video_data, label = batch
        pred = self.forward(video_data)
        loss_function = nn.MSELoss()
        loss = loss_function(pred, label)

        self.log("val_loss", loss)

        return loss
    
    def test_step(self, batch, batch_idx):
        video_data, label = batch
        pred = self.forward(video_data)
        loss_function = nn.MSELoss()
        loss = loss_function(pred, label)

        self.log("test_loss", loss)

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        video_data, label = batch

        return self.forward(video_data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]