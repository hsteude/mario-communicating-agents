import os

# paths
DATA_VERSION = 1
DATA_ROOT_PATH = 'data'
IMGS_SERIES_FOLDER_PATH = os.path.join(DATA_ROOT_PATH,
                                       f'imgs_series_{DATA_VERSION}')
LABELS_TABLE_PATH = os.path.join(DATA_ROOT_PATH,
                                 f'labels_table_{DATA_VERSION}.csv')
LABELS_TABLE_QA_PATH = os.path.join(DATA_ROOT_PATH,
                                    f'labels_table_qa_{DATA_VERSION}.csv')

# Question Answer stuff
MARIO_SPEED = 50
BOX_X_MIN = 300
BOX_X_MAX = 800
ENEMY_SPEED_MIN = 20
ENEMY_SPEED_MAX = 40
PIPE_X_MIN = 900
PIPE_X_MAX = 11000
MARIO_START_X = 190
ENEMY_START_X = 300

# columns and stuff
IMGS_PATH_COL = 'imgs_folder_path'
HIDDEN_STATE_COLS = ['box_x', 'pipe_x', 'enemy_speed']
ANSWER_COLS = ['answer_mario_box', 'answer_mario_pipe', 'answer_mario_enemy']
QUESTION_COL = 'question_mario_speed'

# encoder params
IMG_SIZE = (224, 224)
# LATENT_SPACE_SIZE = 5
# ENC_DR_RATE = 0
# ENC_RNN_HIDDEN_DIM = 6
# ENC_RNN_NUM_LAYERS = 1
# ENC_PRETRAINED = True
# FIXED_CNN_WEIGHTS = True
