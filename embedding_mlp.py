from __future__ import print_function

from keras import losses
from keras.layers import Dense, Input, Embedding, Flatten, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys

from util.dataset import load_data_v1, make_submission


def embedding_mlp(cont_feature_dim):
    day_of_week_input = Input(shape=(1,))
    day_of_week_embedding = Embedding(8, 10, input_length=1)(day_of_week_input)
    day_of_week_embedding = Flatten()(day_of_week_embedding)

    create_hour_input = Input(shape=(1,))
    create_hour_embedding = Embedding(25, 10, input_length=1)(create_hour_input)
    create_hour_embedding = Flatten()(create_hour_embedding)

    poi_points_input = Input(shape=(2,))
    poi_points_embedding = Embedding(1235, 50, input_length=2)(poi_points_input)
    poi_points_embedding = Flatten()(poi_points_embedding)

    continuous_features = Input(shape=(cont_feature_dim,))

    mlp_input = concatenate([day_of_week_embedding, create_hour_embedding, poi_points_embedding, continuous_features])

    mlp_hidden1 = Dense(128, activation='relu')(mlp_input)
    mlp_hidden2 = Dense(64, activation='relu')(mlp_hidden1)
    mlp_hidden3 = Dense(32, activation='relu')(mlp_hidden2)

    mlp_output = Dense(1)(mlp_hidden3)

    model = Model(inputs=[day_of_week_input, create_hour_input, poi_points_input, continuous_features],
                  outputs=mlp_output)
    model.compile(loss=losses.mae, optimizer='adam')

    return model


if __name__ == '__main__':

    submit_flag = True if sys.argv[1] == 'submit' else False

    print('***** Start UAI-CUP-2017 *****')
    print('Loading data ...')
    x_train, y_train, x_dev, y_dev, x_test, mm_scalar = load_data_v1(data_path='data')

    x_train_cont_features = x_train[:, 0:21]
    x_train_day_of_week = x_train[:, 21]
    x_train_create_hour = x_train[:, 22]
    x_train_poi_points = x_train[:, 23:25]

    x_dev_cont_features = x_dev[:, 0:21]
    x_dev_day_of_week = x_dev[:, 21]
    x_dev_create_hour = x_dev[:, 22]
    x_dev_poi_points = x_dev[:, 23:25]

    x_test_cont_features = x_test[:, 0:21]
    x_test_day_of_week = x_test[:, 21]
    x_test_create_hour = x_test[:, 22]
    x_test_poi_points = x_test[:, 23:25]

    print('Training Embedding MLP model ...')
    check_pointer = ModelCheckpoint(filepath='models/embedding_mlp.hdf5', verbose=1, save_best_only=True,
                                    save_weights_only=True)
    early_stopping = EarlyStopping(patience=3)
    csv_logger = CSVLogger('logs/embedding_mlp.log')
    embedding_mlp_model = embedding_mlp(cont_feature_dim=x_train_cont_features.shape[1])
    embedding_mlp_model.fit([x_train_day_of_week, x_train_create_hour, x_train_poi_points, x_train_cont_features],
                            y_train, batch_size=128, epochs=100, verbose=1,
                            validation_data=([x_dev_day_of_week,
                                              x_dev_create_hour,
                                              x_dev_poi_points,
                                              x_dev_cont_features], y_dev),
                            callbacks=[check_pointer, early_stopping, csv_logger])

    if submit_flag:
        print('Generate submission ...')
        embedding_mlp_model.load_weights(filepath='models/embedding_mlp.hdf5')
        results = embedding_mlp_model.predict([x_test_day_of_week,
                                               x_test_create_hour,
                                               x_test_poi_points,
                                               x_dev_cont_features]).reshape(-1, 1)
        make_submission(result_path='submissions', results=results, model_name='Embedding_MLP')

    print('***** End UAI-CUP-2017 *****')
