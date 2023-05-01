import torch

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
import pytorch_lightning as pl
import pandas as pd
from copy import deepcopy
import os

class customdataset(torch.utils.data.Dataset):
    def __init__(self, csv_dataset) -> None:
        super().__init__()

        df_dataset = pd.read_csv(csv_dataset)
        
        self.df_dataset = df_dataset

        side_size = 224
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 224
        fast_num_frames = 32
        slow_num_frames = 8
        sampling_rate = 8
        frames_per_second = 30

        self.fast_transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(fast_num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        self.slow_transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(slow_num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )
        self.fast_clip_duration = (fast_num_frames * sampling_rate)/frames_per_second
        self.slow_clip_duration = (slow_num_frames * sampling_rate)/frames_per_second

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = self.df_dataset.iloc[idx]["paths"]

        label = self.df_dataset.iloc[idx]["label"]
        
        label = torch.tensor(label, dtype=torch.float)
        label = label.unsqueeze(0)
        label = label * 0.01

        video = EncodedVideo.from_path(video_path)
        video_data1 = video.get_clip(start_sec=0, end_sec=10)
        video_data2 = deepcopy(video_data1)

        fast_video_data = self.fast_transform(video_data1)
        slow_video_data = self.slow_transform(video_data2)
        inputs = [slow_video_data["video"], fast_video_data["video"]]

        return inputs, label
    
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        
        self._BATCH_SIZE = batch_size
        self._NUM_WORKERS = num_workers

    def train_dataloader(self):
        train_dataset = customdataset(csv_dataset=os.path.join(os.getcwd(), 'test.csv'))
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )

    def val_dataloader(self):
        val_dataset = customdataset(csv_dataset=os.path.join(os.getcwd(), 'val.csv'))
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )
    def test_dataloader(self):
        test_dataset = customdataset(csv_dataset=os.path.join(os.getcwd(), 'test.csv'))
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            drop_last=True
        )