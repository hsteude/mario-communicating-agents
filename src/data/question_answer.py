import src.constants as const
import numpy as np


class QuestionAndOptimalAnswerGenerator():
    def __init__(self, df, mario_start_x, enemy_start_x):
        np.random.seed(56)
        self.mario_start_x = mario_start_x
        self.enemy_start_x = enemy_start_x
        self.df = df
        # change back to min and max speed if needed
        self.df.loc[:, const.QUESTION_COL] = np.random.uniform(
            const.MARIO_SPEED_QUEST_MIN,
            const.MARIO_SPEED_QUEST_MAX,
            len(self.df)
        )

    def _compute_answer_mario_box(self, mario_speed, box_x):
        distance = box_x - self.mario_start_x
        return distance / mario_speed

    def _compute_answer_mario_pipe(self, mario_speed, pipe_x):
        distance = pipe_x - self.mario_start_x
        return distance / mario_speed

    def _compute_answer_mario_enemy(self, mario_speed, enemy_speed):
        """
        x_m = x_m0 + v_m * t
        x_e = x_e0 + v_e * t
        x_m0 + v_m * t == x_e0 + v_e * t
        t == (x_e0  - x_m0) / (v_m - v_e)
        """
        return (self.enemy_start_x - self.mario_start_x) /\
            (mario_speed - enemy_speed)

    def compute_ansers(self):
        funcs = [self._compute_answer_mario_box,
                 self._compute_answer_mario_pipe,
                 self._compute_answer_mario_enemy]
        in_cols = [(const.QUESTION_COL, const.HIDDEN_STATE_COLS[0]),
                   (const.QUESTION_COL, const.HIDDEN_STATE_COLS[1]),
                   (const.QUESTION_COL, const.HIDDEN_STATE_COLS[2])
                   ]
        for func, in_col, out_col in zip(
                funcs, in_cols, const.ANSWER_COLS):
            self.df.loc[:, out_col] = \
                [func(in1, in2) for in1, in2 in zip(
                    self.df[in_col[0]], self.df[in_col[1]])]

    def run(self):
        self.compute_ansers()
        self.df.to_csv(const.LABELS_TABLE_QA_PATH)
