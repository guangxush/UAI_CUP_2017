from __future__ import print_function

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from matplotlib import pyplot

import sys

from util.dataset import load_data_v2, make_submission


def xgb():
    model = XGBRegressor()
    return model


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test = load_data_v2(data_path='data')

    print('Training XGBoost Regression model ...')

    xgb_model = xgb()

    eval_set = [(x_dev, y_dev)]
    xgb_model.fit(x_train, y_train,
                  early_stopping_rounds=3,
                  eval_metric='mae',
                  eval_set=eval_set,
                  verbose=True)
    plot_importance(xgb_model)
    pyplot.show()

    if submit_flag:
        print('Generate submission ...')
        results = xgb_model.predict(x_test).reshape(-1, 1)
        make_submission(result_path='submissions', results=results, model_name='xgboost')

    print('***** End UAI-CUP-2017 *****')
