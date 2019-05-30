from __future__ import print_function

from keras.layers import Dense, Input, Embedding, Flatten, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import numpy as np

import sys

from util.dataset import load_data_v5, make_submission


def classification_mlp(sample_dim):
    feature_input = Input(shape=(sample_dim,), name='mlp_input')

    x = Dense(512, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim)(feature_input)
    x = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dense(64, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dense(32, kernel_initializer='glorot_uniform', activation='relu')(x)

    classification_output = Dense(1, activation='sigmoid', name='c_output')(x)

    model = Model(inputs=[feature_input],
                  outputs=[classification_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def regression_mlp(sample_dim):
    feature_input = Input(shape=(sample_dim,), name='mlp_input')

    x = Dense(512, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim)(feature_input)
    x = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dense(64, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dense(32, kernel_initializer='glorot_uniform', activation='relu')(x)

    reg_output = Dense(1, name='r_output')(x)

    model = Model(inputs=[feature_input],
                  outputs=[reg_output])
    model.compile(optimizer='adam', loss='mae')

    return model


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train_c, y_train_c, x_train_r, y_train_r, x_test = load_data_v5(data_path='data')

    early_stopping = EarlyStopping(patience=3)
    # print('Training Classification Phase MLP model ...')
    # c_check_pointer = ModelCheckpoint(filepath='models/c_phase_mlp.hdf5', verbose=1, save_best_only=True,
    #                                   save_weights_only=True)
    # c_csv_logger = CSVLogger('logs/c_phase_mlp.log')
    c_model = classification_mlp(sample_dim=x_train_c.shape[1])
    # c_model.fit({'mlp_input': x_train_c},
    #             {'c_output': y_train_c},
    #             validation_split=0.1,
    #             batch_size=128, epochs=50, verbose=1,
    #             callbacks=[c_check_pointer, early_stopping, c_csv_logger])

    print('Training Regression Phase MLP model ...')
    # r_check_pointer = ModelCheckpoint(filepath='models/r_phase_mlp.hdf5', verbose=1, save_best_only=True,
    #                                   save_weights_only=True)
    # r_csv_logger = CSVLogger('logs/r_phase_mlp.log')
    r_model = regression_mlp(sample_dim=x_train_r.shape[1])
    # r_model.fit({'mlp_input': x_train_r},
    #             {'r_output': y_train_r},
    #             validation_split=0.1,
    #             batch_size=128, epochs=50, verbose=1,
    #             callbacks=[r_check_pointer, early_stopping, r_csv_logger])

    if submit_flag:
        print('Generate submission ...')

        c_model.load_weights(filepath='models/c_phase_mlp.hdf5')
        r_model.load_weights(filepath='models/r_phase_mlp.hdf5')
        c_results = c_model.predict([x_test]).reshape(-1, 1)
        r_results = r_model.predict([x_test]).reshape(-1, 1)

        results = list()
        for item in zip(c_results, r_results):
            if item[0][0] > 0.5:
                results.append(item[1][0])
            else:
                results.append(1.0)
        results = np.array(results).reshape(-1, 1)
        make_submission(result_path='submissions', results=results, model_name='Two_Phase_MLP')

    print('***** End UAI-CUP-2017 *****')
