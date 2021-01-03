from mario_game.source import tools
import numpy as np
from mario_game.source import constants as c
from src import constants as const
from mario_game.source.states import main_menu, load_screen, level
import os
import argparse


def main(num_img, labels_table_path, imgs_folder_path):

    # random hidden states
    box_x = np.random.choice(range(300, 800))
    pipe_x = np.random.choice(range(900, 1100))
    enemy_speed = np.random.choice(range(25, 50))

    game = tools.Control()
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.LEVEL: level.Level(box_x, pipe_x, enemy_speed),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.TIME_OUT: load_screen.TimeOut()}
    game.setup_states(state_dict, c.MAIN_MENU)
    game.main(num_img, imgs_folder_path)

    # write paths and random state vals to table
    header_row = 'imgs_folder_path,box_x,pipe_x,enemy_speed\n'
    labels_row = f'{imgs_folder_path},{box_x},{pipe_x},{enemy_speed}\n'
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
