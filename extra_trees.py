from __future__ import print_function

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error


import sys

from util.dataset import load_data_v2, make_submission

import numpy as np

seed = 13
np.random.seed(seed)


def extra_trees():
    etr_model = ExtraTreesRegressor(criterion='mae', n_jobs=1, verbose=2)
    return etr_model


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test = load_data_v2(data_path='data')

    print('Training Extra Trees Regression model ...')

    etr = extra_trees()
    etr.fit(x_train, y_train)
    etr_y_train = etr.predict(x_train).reshape(-1, 1)
    etr_y_dev = etr.predict(x_dev).reshape(-1, 1)

    train_mae = mean_absolute_error(y_train, etr_y_train)
    print('Train MAE:', train_mae)

    dev_mae = mean_absolute_error(y_dev, etr_y_dev)
    print('Dev MAE:', dev_mae)

    if submit_flag:
        print('Generate submission ...')
        results = etr.predict(x_test).reshape(-1, 1)
        make_submission(result_path='submissions', results=results, model_name='Extra_Trees')

    print('***** End UAI-CUP-2017 *****')
