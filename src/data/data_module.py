from pytorch_lightning.core.datamodule import LightningDataModule
import src.constants as const
from src.data.dataset import (VideoLabelDataset,
                              VideoFolderPathToTensor,
                              VideoResize)
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader


class VideoDataModule(LightningDataModule):
    def __init__(self, validdation_split, batch_size, dl_num_workers,
                 *args, **kwargs):
        self.validdation_split = validdation_split
        self.batch_size = batch_size
        self.dl_num_workers = dl_num_workers
        super().__init__()

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # data set and loader related
        dataset_full = VideoLabelDataset(
            const.LABELS_TABLE_QA_PATH,
            img_transform=torchvision.transforms.Compose([
                VideoFolderPathToTensor(),
                VideoResize(const.IMG_SIZE)]))
        dataset_size = len(dataset_full)
        len_val = int(np.floor(dataset_size * self.validdation_split))
        len_train = dataset_size - len_val
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            dataset=dataset_full, lengths=[len_train, len_val],
            generator=torch.Generator())

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          num_workers=self.dl_num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          num_workers=self.dl_num_workers,
                          pin_memory=True)

