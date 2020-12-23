"""inspired by https://github.com/YuxinZhaozyx/pytorch-VideoDataset"""
import torch
import torchvision
import numpy as np
import PIL
import collections
import cv2
import os


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
        interpolation (int, optional): Desired interpolation. Default is 'PIL.Image.BILINEAR'
    """

    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        assert isinstance(size, collections.Iterable) and len(size) == 2
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
        C, L, H, W = video.size()
        rescaled_video = torch.FloatTensor(C, L, h, w)

        # use torchvision implemention to resize video frames
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(self.size, self.interpolation),
            torchvision.transforms.ToTensor(),
        ])

        for l in range(L):
            frame = video[:, l, :, :]
            frame = transform(frame)
            rescaled_video[:, l, :, :] = frame

        return rescaled_video

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class VideoRandomCrop(object):
    """ Crop the given Video Tensor (C x L x H x W) at a random location.
    Args:
        size (sequence): Desired output size like (h, w).
    """

    def __init__(self, size):
        assert len(size) == 2
        self.size = size

    def __call__(self, video):
        """ 
        Args:
            video (torch.Tensor): Video (C x L x H x W) to be cropped.
        Returns:
            torch.Tensor: Cropped video (C x L x h x w).
        """

        H, W = video.size()[2:]
        h, w = self.size
        assert H >= h and W >= w

        top = np.random.randint(0, H - h)
        left = np.random.randint(0, W - w)

        video = video[:, :, top: top + h, left: left + w]

        return video

