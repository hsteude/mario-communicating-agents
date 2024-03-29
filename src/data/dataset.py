import os

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms.functional import InterpolationMode

import src.constants as const


# cv2.setNumThreads(0)


class VideoLabelDataset(Dataset):
    """Write me!"""

    def __init__(self, csv_file, img_transform=None):
        self.dataframe = pd.read_csv(csv_file)
        scaler = StandardScaler()
        scaling_cols = const.ANSWER_COLS + const.HIDDEN_STATE_COLS \
            + [const.QUESTION_COL]
        self.dataframe.loc[:, scaling_cols] = scaler.fit_transform(
            self.dataframe[scaling_cols])
        self.img_transform = img_transform

    def __len__(self):
        """Size of dataset
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """Get one sample (including questions and answers)"""
        video_path = self.dataframe.iloc[index].imgs_folder_path
        answers = self.dataframe.loc[
            index, const.ANSWER_COLS].values.astype(np.float32)
        hidden_states = self.dataframe.loc[
            index, const.HIDDEN_STATE_COLS].values.astype(np.float32)
        questions = self.dataframe.loc[index, const.QUESTION_COL].astype(np.float32)
        if self.img_transform:
            video = self.img_transform(video_path)
        return video, answers, hidden_states, questions, video_path


class VideoFolderPathToTensor(object):
    """Load video at given folder path to torch.Tensor (C x L x H x W)
        It can be composed with torchvision.transforms.Compose().
    """

    def __call__(self, path):
        """
        Args:
            path (str): path of video folder.

        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
        # get video properity
        frames_path = sorted([os.path.join(path, f) for f in os.listdir(path)
                              if os.path.isfile(os.path.join(path, f))])
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)

        time_len = num_frames

        frames = torch.FloatTensor(channels, time_len, height, width)

        # load the video to tensor
        for index in range(time_len):
            # read frame
            frame = cv2.imread(frames_path[index])
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames[:, index, :, :] = frame.float()
        frames /= 255
        return frames


class VideoResize(object):
    """Resize video tensor (C x L x H x W) to (C x L x h x w)

    Args:
        size (sequence): Desired output size. size is a sequence like (H, W),
            output size will matched to this.
        interpolation (int, optional): Desired interpolation. Default is
        'PIL.Image.BILINEAR'
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video):
        """
        Args:
            video (torch.Tensor): Video to be scaled (C x L x H x W)

        Returns:
            torch.Tensor: Rescaled video (C x L x h x w)
        """

        h, w = self.size
        C, L, _, _ = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)

        # use torchvision implemention to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for len in range(L):
            frame = video[:, len, :, :]
            frame = transform(frame)
            rescaled_video[:, len, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


if __name__ == '__main__':
    # test for VideoLabelDataset
    labels_path = './data/labels_table_qa.csv'
    dataset = VideoLabelDataset(
        labels_path,
        img_transform=torchvision.transforms.Compose([
            VideoFolderPathToTensor(),
            VideoResize((224, 224))]))
    idx = 100
    video, answers, hidden_states, questions, video_path = dataset[idx]
    frame0 = torchvision.transforms.ToPILImage()(video[:, 0, :, :])
    frameN = torchvision.transforms.ToPILImage()(video[:, 10, :, :])
    frame0.show()
    frameN.show()
    print(f'Question {idx}: {questions}')
    print(f'Answer {idx}: {answers}')
