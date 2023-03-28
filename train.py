import pytorch_lightning as pl
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
import os

warnings.filterwarnings('ignore')
from dataset import CustomDataModule
from model import VideoClassificationLightningModule
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set Model', add_help=False)
    parser.add_argument('--batch_size', default=1, type = int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--model_name',default='slow_r50', type=str)
    return parser

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Script', parents=[get_args_parser()])
    args = parser.parse_args()

    checkpoint_root = './checkpoint'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_root, save_top_k=1, monitor='val_loss', mode = 'min', save_last=True)

    model = VideoClassificationLightningModule(model_name = args.model_name, batch_size = args.batch_size)
    data_module = CustomDataModule(batch_size = args.batch_size, num_workers = args.num_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.device, max_epochs=args.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)