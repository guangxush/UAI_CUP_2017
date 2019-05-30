from __future__ import print_function

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


import sys

from util.dataset import load_data_v2, make_submission

import numpy as np

seed = 13
np.random.seed(seed)


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test = load_data_v2(data_path='data')

    print('Training KNN Regression model ...')

    knn_reg = KNeighborsRegressor()
    knn_reg.fit(x_train, y_train)
    knn_y_train = knn_reg.predict(x_train).reshape(-1, 1)
    knn_y_dev = knn_reg.predict(x_dev).reshape(-1, 1)

    train_mae = mean_absolute_error(y_train, knn_y_train)
    print('Train MAE:', train_mae)

    dev_mae = mean_absolute_error(y_dev, knn_y_dev)
    print('Dev MAE:', dev_mae)

    if submit_flag:
        print('Generate submission ...')
        results = knn_reg.predict(x_test).reshape(-1, 1)
        make_submission(result_path='submissions', results=results, model_name='KNN')

    print('***** End UAI-CUP-2017 *****')
