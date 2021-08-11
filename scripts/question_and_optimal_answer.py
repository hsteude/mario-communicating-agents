import pandas as pd

import src.constants as const
from src.data.question_answer import QuestionAndOptimalAnswerGenerator
import torch
import torchvision

df = pd.read_csv(const.LABELS_TABLE_PATH)
# from src.data.dataset import (VideoLabelDataset,
                              # VideoFolderPathToTensor,
                              # VideoResize)
# dataset = VideoLabelDataset(
            # const.LABELS_TABLE_QA_PATH,
            # img_transform=torchvision.transforms.Compose([
                # VideoFolderPathToTensor(),
                # VideoResize(const.IMG_SIZE)]))
# df = dataset.dataframe
qaoag = QuestionAndOptimalAnswerGenerator(df, const.MARIO_START_X, const.ENEMY_START_X)
qaoag.run()
