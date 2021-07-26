import pandas as pd

import src.constants as const
from src.data.question_answer import QuestionAndOptimalAnswerGenerator

df = pd.read_csv(const.LABELS_TABLE_PATH)
qaoag = QuestionAndOptimalAnswerGenerator(df, const.MARIO_START_X, const.ENEMY_START_X)
qaoag.run()
