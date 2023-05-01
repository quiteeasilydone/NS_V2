import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl
import sklearn.metrics as mt

class attention_mask(nn.Module):
    def __init__(self, transformer, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.9, device = 'cuda'):
        super().__init__()
        self._device = device
        self.model = transformer
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def rollout(self, attentions, discard_ratio, head_fusion):
        result = torch.eye(attentions[0].size(-1), device=self._device)
        with torch.no_grad():
            for attention in attentions:
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1), device=self._device)
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
        mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width)
        mask = mask / torch.max(mask)
        
        return mask
    
    def forward(self, x):
        b,c,h,w = x.size()
        self.attentions = []
        with torch.no_grad():
            output = self.model(x)
        mask = self.rollout(self.attentions, self.discard_ratio, self.head_fusion)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, (h,w), mode='bilinear')

        return mask

class Videomodel(pl.LightningModule):
    def __init__(self, backbone, transformer, device = 'cuda'):
        super().__init__()
        self._device = device
        self.attention_mask_generator = attention_mask(transformer, device=self._device)
        self.backbone = backbone
    
    def forward(self, x):
        input_x = []
        for inputs in x:
            attention_input = torch.einsum('bcfhw -> bfchw', inputs)
            b,f,c,h,w = attention_input.size()
            batch_masks = []
            for batch in range(b):
                masks = []
                for frame in range(f):
                    input_img = attention_input[batch][frame].unsqueeze(0)
                    attention = self.attention_mask_generator(input_img)
                    attention = attention.squeeze()
                    masks.append(attention)
                tensor_masks = torch.stack(masks, dim = 0)
                batch_masks.append(tensor_masks)
            tensor_batch_masks = torch.stack(batch_masks, dim = 0)
            tensor_batch_masks = tensor_batch_masks.unsqueeze(dim = 1)
            input_attention = torch.cat((inputs, tensor_batch_masks), dim=1)
            input_x.append(input_attention)
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
        R2_score = mt.r2_score(label, pred)
        MSE_error = mt.mean_squared_error(label, pred)
        MAE_error = mt.mean_absolute_error(label, pred)
        
        self.log("MAE_error", MAE_error)
        self.log("MSE_error", MSE_error)
        self.log("R2_score", R2_score)
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        video_data, label = batch

        return self.forward(video_data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], [scheduler]