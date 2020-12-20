import pandas as pd
import src.constants as const

df = pd.read_csv(const.LABELS_TABLE_PATH)


class OptimalAnswerGenerator():
    def __init__(self, df, mario_speed, mario_start_x, enemy_start_x):
        self.mario_start_x = mario_start_x
        self.enemy_start_x = enemy_start_x
        self.mario_speed = mario_speed
        self.df = df

    def _compute_answer_box(self, box_x):
        distance = box_x - self.mario_start_x
        return distance / self.mario_speed

    def _compute_answer_pipe(self, pipe_x):
        distance = pipe_x - self.mario_start_x
        return distance / self.mario_speed

    def _compute_anser_enemy(self, enemy_speed):
        """
        x_m = x_m0 + v_m * t
        x_e = x_e0 + v_e * t
        x_m0 + v_m * t == x_e0 + v_e * t
        t == (x_e0  - x_m0) / (v_m - v_e)
        """
        return (self.enemy_start_x - self.mario_start_x) /\
            (self.mario_speed - enemy_speed)

    def compute_ansers():
        

    
