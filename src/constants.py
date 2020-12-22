import numpy as np
import os

# paths
DATA_ROOT_PATH = 'data'
IMGS_SERIES_FOLDER_PATH = os.path.join(DATA_ROOT_PATH, 'imgs_series')
LABELS_TABLE_PATH = os.path.join(DATA_ROOT_PATH, 'labels_table.csv')
LABELS_TABLE_QA_PATH = os.path.join(DATA_ROOT_PATH, 'labels_table_qa.csv')


# random hidden states
box_x = np.random.uniform(low=300, high=800, size=1)
pipe_x = np.random.uniform(low=900, high=1100, size=1)
enemy_speed = np.random.choice(range(25, 50))

# random question parameters (marios speed)
MARIO_SPEED_MIN = 50
MARIO_SPEED_MAX = 100

# columns and stuff
IMGS_PATH_COL = 'imgs_folder_path'
HIDDEN_STATE_COLS = ['box_x', 'pipe_x', 'enemy_speed']
QUESTION_COL = 'mario_speed'
ANSWER_COLS = ['answer_box', 'answer_pipe', 'answer_enemy']
