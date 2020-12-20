import pygame as pg
from mario_game.source import setup, tools
from mario_game.source import constants as c
from mario_game.source.states import main_menu, load_screen, level
import os
import argparse


def main(num_img, run):
    
    game = tools.Control()
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.LEVEL: level.Level(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.TIME_OUT: load_screen.TimeOut()}
    igs_path = os.path.join('data', 'test', f'{run:05d}')
    game.setup_states(state_dict, c.MAIN_MENU)
    game.main(num_img, igs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int)
    args = parser.parse_args()

    num_img = 10
    main(num_img, args.run)
