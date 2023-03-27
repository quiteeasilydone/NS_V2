import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import warnings
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
import os
import pandas as pd

warnings.filterwarnings('ignore')

from detr_model import attention_detr
from slow_model import create_model
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set Model', add_help=False)
    parser.add_argument('--batch_size', default=1, type = int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--strategy', default='ddp', type=str)
    return parser

class test_model(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.attention_weights = attention_detr(self.batch_size)
        self.backbone = create_model()
    
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
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_dataset) -> None:
        super().__init__()

        df_dataset = pd.read_csv(csv_dataset)
        
        self.df_dataset = df_dataset

        side_size = 1024
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 1024
        num_frames = 8
        sampling_rate = 8
        frames_per_second = 30

        self.transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        self.clip_duration = (num_frames * sampling_rate)/frames_per_second

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = self.df_dataset.iloc[idx]["paths"]

        label = self.df_dataset.iloc[idx]["label"]
        
        label = torch.tensor(label, dtype=torch.float)
        label = label.unsqueeze(0)
        label = label * 0.01

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=10)
        video_data = self.transform(video_data)
        inputs = video_data["video"]

        return inputs, label
    
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers

    def train_dataloader(self):
        train_dataset = CustomDataset(csv_dataset=os.path.join(os.getcwd(), 'test.csv'))
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )

    def val_dataloader(self):
        val_dataset = CustomDataset(csv_dataset=os.path.join(os.getcwd(), 'val.csv'))
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )
    def test_dataloader(self):
        test_dataset = CustomDataset(csv_dataset=os.path.join(os.getcwd(), 'test.csv'))
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Script', parents=[get_args_parser()])
    args = parser.parse_args()

    checkpoint_root = './checkpoint'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_root, save_top_k=1, monitor='val_loss', mode = 'min', save_last=True)

    model = test_model(batch_size = args.batch_size)
    data_module = CustomDataModule(batch_size = args.batch_size, num_workers = args.num_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.device, max_epochs=args.epochs, strategy=args.strategy, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)