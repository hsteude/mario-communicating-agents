import src.constants as const
import numpy as np


class QuestionAndOptimalAnswerGenerator():
    def __init__(self, df, mario_start_x, enemy_start_x):
        self.mario_start_x = mario_start_x
        self.enemy_start_x = enemy_start_x
        self.df = df

    def _compute_answer_box(self, mario_speed, box_x):
        distance = box_x - self.mario_start_x
        return distance / mario_speed

    def _compute_answer_pipe(self, mario_speed, pipe_x):
        distance = pipe_x - self.mario_start_x
        return distance / mario_speed

    def _compute_anser_enemy(self, mario_speed, enemy_speed):
        """
        x_m = x_m0 + v_m * t
        x_e = x_e0 + v_e * t
        x_m0 + v_m * t == x_e0 + v_e * t
        t == (x_e0  - x_m0) / (v_m - v_e)
        """
        return (self.enemy_start_x - self.mario_start_x) /\
            (mario_speed - enemy_speed)

    def compute_questions(self):
        self.df.loc[:, const.QUESTION_COL] = np.random.uniform(
            low=const.MARIO_SPEED_MIN, high=const.MARIO_SPEED_MAX,
            size=len(self.df))

    def compute_ansers(self):
        funcs = [self._compute_answer_box, self._compute_answer_pipe,
                 self._compute_anser_enemy]
        for func, in_col, out_col in zip(
                funcs, const.HIDDEN_STATE_COLS, const.ANSWER_COLS):
            self.df.loc[:, out_col] = \
                [func(ms, param) for ms, param in zip(
                    self.df.mario_speed, self.df[in_col])]

    def run(self):
        self.compute_questions()
        self.compute_ansers()
        self.df.to_csv(const.LABELS_TABLE_QA_PATH)
