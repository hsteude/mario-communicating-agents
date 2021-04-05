import numpy as np
import itertools
import pandas as pd


class FormulaFeatureGenerator():
    def __init__(self, exponents: list = [-1, -.5, 0, 1, 2]):
        self.exponents = exponents
        self.small_number = 1e-2

    def fit(self, x: np.array):
        self.var_list = np.arange(x.shape[1])
        var_exps_lists = [[(v, e) for e in self.exponents]
                          for v in self.var_list]
        self.combinations_exps = list(itertools.product(*var_exps_lists))
        combinations_frac = list(itertools.combinations(self.var_list, 2, ))
        self.combinations_frac = combinations_frac \
            + [(e[1], e[0]) for e in combinations_frac]

    def get_feature_names(self):
        assert hasattr(self, 'combinations_exps'), 'call fit method first'
        feature_names_exp = ['*'.join([f'x_{c[i][0]}^({c[i][1]})'
                                       if c[i][1] != 0 else '1'
                                       for i in self.var_list])
                             for c in self.combinations_exps]
        feature_names_frac = [
            f'1/(x_{c[0]} - x_{c[1]})' for c in self.combinations_frac]
        return feature_names_exp + feature_names_frac

    def _create_single_exp_feature(self, x: np.array, tup: tuple):
        x[x == 0] = self.small_number
        return [np.prod([np.real(float(x[i, j])**ei) for j, ei in tup])
                for i in range(x.shape[0])]

    def _create_single_frac_feature(self, x: np.array, tup: tuple):
        x[x == 0] = self.small_number
        return 1 / (x[:, tup[0]] - x[:, tup[1]])

    def transform(self, x: np.array):
        x_new = np.zeros(
            (x.shape[0], len(self.combinations_exps + self.combinations_frac)))
        for i, tup in enumerate(self.combinations_exps):
            x_new[:, i] = self._create_single_exp_feature(x, tup)
        for i, tup in enumerate(self.combinations_frac):
            x_new[:, i + len(self.combinations_exps)] = \
                self._create_single_frac_feature(x, tup)
        return x_new

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def get_data_frame(self, x_trans):
        return pd.DataFrame(x_trans, columns=self.get_feature_names())




if __name__ == '__main__':
    breakpoint()
    x = np.arange(16).reshape(-1, 4)
    ffg = FormulaFeatureGenerator()
    ffg.fit(x)
    x_trans = ffg.transform(x)
