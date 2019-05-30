from __future__ import print_function

from keras.layers import Dense, Input, Embedding, Flatten, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import sys

from util.dataset import load_data_v4, make_submission


def multi_task_mlp(sample_dim):
    feature_input = Input(shape=(sample_dim,), name='mlp_input')

    x = Dense(512, kernel_initializer='glorot_uniform', activation='relu', input_dim=sample_dim)(feature_input)
    x = Dense(128, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dense(64, kernel_initializer='glorot_uniform', activation='relu')(x)
    x = Dense(32, kernel_initializer='glorot_uniform', activation='relu')(x)

    reg_output = Dense(1, name='r_output')(x)
    classification_output = Dense(1, activation='sigmoid', name='c_output')(x)

    model = Model(inputs=[feature_input],
                  outputs=[reg_output, classification_output])
    model.compile(optimizer='adam',
                  loss={'r_output': 'mae', 'c_output': 'binary_crossentropy'},
                  loss_weights={'r_output': 1., 'c_output': 1.})

    return model


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train, y_train_c, y_train_r, x_dev, y_dev_c, y_dev_r, x_test = load_data_v4(data_path='data')

    print('Training Multi-Task MLP model ...')
    check_pointer = ModelCheckpoint(filepath='models/multi_task_mlp.hdf5', verbose=1, save_best_only=True,
                                    save_weights_only=True)
    early_stopping = EarlyStopping(patience=3)
    csv_logger = CSVLogger('logs/multi_task_mlp.log')
    mt_model = multi_task_mlp(sample_dim=x_train.shape[1])
    mt_model.fit({'mlp_input': x_train},
                 {'r_output': y_train_r, 'c_output': y_train_c},
                 validation_data=([x_dev], [y_dev_r, y_dev_c]),
                 batch_size=128, epochs=50, verbose=1,
                 callbacks=[check_pointer, early_stopping, csv_logger])

    if submit_flag:
        print('Generate submission ...')
        mt_model.load_weights(filepath='models/multi_task_mlp.hdf5')
        results = mt_model.predict([x_test])[0].reshape(-1, 1)
        make_submission(result_path='submissions', results=results, model_name='Multi_Task_MLP')

    print('***** End UAI-CUP-2017 *****')
