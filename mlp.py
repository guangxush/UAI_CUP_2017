from __future__ import print_function

from keras import losses
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from sklearn.metrics import mean_absolute_error

from util.dataset import load_data_v9, make_submission

import numpy as np

import sys


def mlp(sample_dim):
    model = Sequential()
    model.add(Dense(512, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim))
    model.add(Dense(128, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1))
    model.compile(loss=losses.mae, optimizer='adam')
    return model


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test = load_data_v9(data_path='data')

    print('Training MLP model ...')
    check_pointer = ModelCheckpoint(filepath='models/mlp.hdf5', verbose=1, save_best_only=True,
                                    save_weights_only=True)
    early_stopping = EarlyStopping(patience=10)
    csv_logger = CSVLogger('logs/mlp.log')
    mlp_model = mlp(sample_dim=x_train.shape[1])
    mlp_model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_dev, y_dev),
                  callbacks=[check_pointer, early_stopping, csv_logger])

    if submit_flag:
        print('Generate submission ...')
        mlp_model.load_weights(filepath='models/mlp.hdf5')
        results = mlp_model.predict(x_test).reshape(-1, 1)
        results_dev = mlp_model.predict(x_dev).reshape(-1, 1)
        results_dev_new = []
        for item in results_dev:
            item_value = item[0]
            item_value = np.round(item_value)
            results_dev_new.append(item_value)
        results_dev_new = np.asarray(results_dev_new).reshape(-1, 1)
        print('Dev MAE:', mean_absolute_error(y_dev, results_dev_new))
        make_submission(result_path='submissions', results=results, model_name='MLP')

    print('***** End UAI-CUP-2017 *****')
