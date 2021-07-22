import argparse
import os

import numpy as np

from mario_game.source import tools
from mario_game.source import constants as c
from mario_game.source.states import level, load_screen, main_menu
from src import constants as const


def main(num_img, labels_table_path, imgs_folder_path):

    # random hidden states
    box_x = np.random.choice(range(const.BOX_X_MIN, const.BOX_X_MAX))
    pipe_x = np.random.choice(range(const.PIPE_X_MIN, const.PIPE_X_MAX))
    enemy_speed = np.random.choice(
        range(const.ENEMY_SPEED_MIN, const.ENEMY_SPEED_MAX))
    mario_speed = const.MARIO_SPEED_OBS

    game = tools.Control()
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.LEVEL: level.Level(
                      box_x, pipe_x, enemy_speed, mario_speed),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.TIME_OUT: load_screen.TimeOut()}
    game.setup_states(state_dict, c.MAIN_MENU)
    game.main(num_img, imgs_folder_path)

    # write paths and random state vals to table
    header_row = 'imgs_folder_path,box_x,pipe_x,enemy_speed,mario_speed\n'
    labels_row = f'{imgs_folder_path},{box_x},{pipe_x},'\
        f'{enemy_speed},{mario_speed}\n'
    if not os.path.isfile(labels_table_path):
        with open(labels_table_path, 'a') as fd:
            fd.write(header_row)
    with open(labels_table_path, 'a') as fd:
        fd.write(labels_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--num_imgs', type=int)
    args = parser.parse_args()

    num_img = args.num_imgs
    run = args.run
    imgs_folder_path = os.path.join(
        const.IMGS_SERIES_FOLDER_PATH, f'{run:05d}')

    main(num_img, const.LABELS_TABLE_PATH, imgs_folder_path)
