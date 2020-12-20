import numpy as np
import os

# paths
DATA_ROOT_PATH = 'data'
IMGS_SERIES_FOLDER_PATH = os.path.join(DATA_ROOT_PATH, 'imgs_series')
LABELS_TABLE_PATH = os.path.join(DATA_ROOT_PATH, 'labels_table.csv')


# random hidden states
box_x = np.random.uniform(low=300, high=800, size=1)
pipe_x = np.random.uniform(low=900, high=1100, size=1)
enemy_speed = np.random.choice(range(25, 50))

# 
HIDDEN_STATE_COLS = ['imgs_folder_path', 'box_x', 'pipe_x', 'enemy_speed']
OPT_ANSWER_SUFFIX = '_opt_answer'


