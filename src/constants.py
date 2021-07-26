import os

# paths
DATA_VERSION = 2
DATA_ROOT_PATH = 'data'
IMGS_SERIES_FOLDER_PATH = os.path.join(DATA_ROOT_PATH,
                                       f'imgs_series_{DATA_VERSION}')
LABELS_TABLE_PATH = os.path.join(DATA_ROOT_PATH,
                                 f'labels_table_{DATA_VERSION}.csv')
LABELS_TABLE_QA_PATH = os.path.join(DATA_ROOT_PATH,
                                    f'labels_table_qa_{DATA_VERSION}.csv')

# Question Answer stuff
MARIO_SPEED_OBS_MIN = 45
MARIO_SPEED_OBS_MAX = 60
BOX_X_MIN = 300
BOX_X_MAX = 800
ENEMY_SPEED_MIN = 25
ENEMY_SPEED_MAX = 40
PIPE_X_MIN = 900
PIPE_X_MAX = 1100
MARIO_START_X = 190
ENEMY_START_X = 300
MARIO_SPEED_QUEST_MIN = 1 # stuff is normalized at the stage of computing optimal answers
MARIO_SPEED_QUEST_MAX = 2 # stuff is normalized at the stage of computing optimal answers


# columns and stuff
IMGS_PATH_COL = 'imgs_folder_path'
HIDDEN_STATE_COLS = ['box_x', 'pipe_x', 'mario_speed', 'enemy_speed']
ANSWER_COLS = ['answer_mario_box', 'answer_enemy_pipe', 'answer_coin_pipe', 'answer_mario_enemy']
#z QUESTION_COL = 'question_mario_speed'

# encoder params
IMG_SIZE = (224, 224)
