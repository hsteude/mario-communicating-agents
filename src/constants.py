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
MARIO_MIN_SPEED = 50
MARIO_MAX_SPEED = 70
#TODO: other vars for experiement.py also define here!!

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
