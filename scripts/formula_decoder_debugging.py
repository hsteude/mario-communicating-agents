import torch
import torchvision
import src.constants as const
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from src.data.dataset import (VideoLabelDataset,
                              VideoFolderPathToTensor,
                              VideoResize)
import plotly
import numpy as np
import pandas as pd
import yaml
import os
from src.model.agents import FormulaFeatureGenerator, FormulaDecoder


dataset = VideoLabelDataset(
            const.LABELS_TABLE_QA_PATH,
            img_transform=None)

dataloader = DataLoader(dataset, batch_size=100)
_, _, hidden_states, _  = iter(dataloader).next()
breakpoint()

ffg = FormulaFeatureGenerator()
X_ = ffg(hidden_states)

fd = FormulaDecoder(X_.shape[1], 1)
fd(X_)




