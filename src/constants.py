import os

# paths
DATA_ROOT_PATH = 'data'
IMGS_SERIES_FOLDER_PATH = os.path.join(DATA_ROOT_PATH, 'imgs_series')
LABELS_TABLE_PATH = os.path.join(DATA_ROOT_PATH, 'labels_table.csv')
LABELS_TABLE_QA_PATH = os.path.join(DATA_ROOT_PATH, 'labels_table_qa.csv')

# Question Answer stuff
MARIO_TARGET = 10000

# columns and stuff
IMGS_PATH_COL = 'imgs_folder_path'
HIDDEN_STATE_COLS = ['box_x', 'pipe_x', 'enemy_speed', 'mario_speed']
ANSWER_COLS = ['answer_mario_box', 'answer_enemy_pipe', 'answer_mario_enemy']

# Training params
BATCH_SIZE = 1
NUM_DL_WORKERS = 12
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.05

# encoder params
IMG_SIZE = (224, 224)
LATENT_SPACE_SIZE = 5
ENC_DR_RATE = 0
ENC_RNN_HIDDEN_DIM = 6
ENC_RNN_NUM_LAYERS = 1
ENC_PRETRAINED = True
FIXED_CNN_WEIGHTS = True
