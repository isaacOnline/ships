import sys
import os
import time
from config import config
import atexit
import json

os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import pandas as pd


from utils import ProcessorManager, TestArgParser

# These need to come before tensorflow is imported so that if we're using CPU we can unregister the GPUs before tf
# imports them.
parser = TestArgParser()
args = parser.parse()
manager = ProcessorManager(debug=args.debug)
manager.open()

import mlflow

if args.debug:
    mlflow.set_experiment(experiment_name='Ships Debugging')


import tensorflow as tf
from mlflow import log_param, log_artifact, log_metric, log_dict
from tensorflow.keras.callbacks import EarlyStopping

import models
import loading
from loading.data_loader import DataLoader
import utils


def save_predictions_and_errors(predictions, errors, args, which):
    """
    Save the predictions and errors from a model as mlflow artifacts

    :param predictions: List of predictions made by the model
    :param errors: List of errors for the model's predictions
    :param args: argparse.Namespace specifying the model
    :param which: One of 'test' or 'validation'
    :return:
    """
    if args.model_type in ['iterative', 'attention_seq2seq', 'transformer']:
        for i in range(1, 4):
            np.savetxt(f'{which}_haversine_error_{i}_hour.csv',
                       errors[i - 1], delimiter=',')
            log_artifact(f'{which}_haversine_error_{i}_hour.csv')

            np.savetxt(f'{which}_predictions_hour_{i}.csv',
                       predictions[i-1], delimiter=',')
            log_artifact(f'{which}_predictions_hour_{i}.csv')


    elif args.model_type in ['long_term', 'long_term_fusion']:
        np.savetxt(f'{which}_haversine_error_{args.hours_out}_hour.csv',
                   errors[0], delimiter=',')
        log_artifact(f'{which}_haversine_error_{args.hours_out}_hour.csv')

        np.savetxt(f'{which}_predictions_hour_{args.hours_out}.csv',
                   predictions[0], delimiter=',')
        log_artifact(f'{which}_predictions_hour_{args.hours_out}.csv')



if __name__ == '__main__':
    # Parse command line arguments

    start_ts = time.time()

    utils.set_seed(args.seed)

    loader = DataLoader(config, args, conserve_memory=True)
    train_Y_labels = loader.load_set('train', 'train', 'y')
    train_X = loader.load_set('train', 'train', 'x')

    valid_Y_labels = loader.load_set('valid', 'train', 'y')
    valid_X = loader.load_set('valid', 'train', 'x')

    if config.logging:
        # Log basic features of run
        loader.run_config.save_to_dir('run_config', register_with_mlflow=True)
        log_param('host', config.host)
        log_param('seed', args.seed)
        log_param('processor', manager.device())
        log_param('lat_1',config.dataset_config.lat_1)
        log_param('lat_2',config.dataset_config.lat_2)
        log_param('lon_1',config.dataset_config.lon_1)
        log_param('lon_2',config.dataset_config.lon_2)
        log_param('start_year', config.start_year)
        log_param('end_year', config.end_year)
        log_param('batch_size', args.batch_size)
        log_param('learning_rate', args.learning_rate)
        log_param('layer_type', args.layer_type)
        log_param('direction', args.direction)
        log_param('number_of_rnn_layers', args.number_of_rnn_layers)
        log_param('rnn_layer_size', args.rnn_layer_size)
        log_param('number_of_dense_layers', args.number_of_dense_layers)
        log_param('dense_layer_size', args.dense_layer_size)
        log_param('weather',args.weather)
        log_param('distance_traveled', args.distance_traveled)
        log_param('sog_cog', args.sog_cog)
        log_param('dataset_name',config.dataset_config.dataset_name)
        log_param('rnn_to_dense_connection', args.rnn_to_dense_connection)
        if args.model_type in ['long_term','long_term_fusion']:
            log_param('regularization', args.regularization)
            log_param('regularization_application',args.regularization_application)
            log_param('regularization_coefficient',args.regularization_coefficient)
            log_param('hours_out', args.hours_out)
            if args.model_type == 'long_term_fusion':
                log_param('number_of_fusion_weather_layers', args.number_of_fusion_weather_layers)
                log_param('fusion_layer_structure', args.fusion_layer_structure)
                log_param('length_of_history',args.length_of_history)
                if args.fusion_layer_structure == 'convolutions':
                    log_param('output_feature_size', args.output_feature_size)
                    log_param('conv_kernel_size', args.conv_kernel_size)
                    log_param('conv_stride_size', args.conv_stride_size)
                    log_param('pool_size', args.pool_size)

        if not args.debug:
            stdout_path = os.path.join(mlflow.get_artifact_uri(), 'stdout.txt').replace('file://', '')
            stderr_path = os.path.join(mlflow.get_artifact_uri(), 'stderr.txt').replace('file://', '')
            sys.stdout = open(stdout_path, 'w', 1)
            sys.stderr = open(stderr_path, 'w', 1)

            def exit_handler():
                sys.stdout.close()
                sys.stderr.close()

            atexit.register(exit_handler)


    # Use the device that this run has been assigned by the manager
    with tf.device(manager.device()):
        # Define model
        if args.model_type == 'iterative':
            runner = models.RNNModelRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                number_of_dense_layers=args.number_of_dense_layers,
                dense_layer_size=args.dense_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                input_num_features=int(train_X.shape[2]),
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate = args.learning_rate,
                loss=args.loss,
                rnn_to_dense_connection=args.rnn_to_dense_connection
            )
        elif args.model_type == 'attention_seq2seq':
            runner = models.Seq2SeqRNNAttentionRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                output_ts_length=int(train_Y_labels.shape[1]),
                input_num_features=int(train_X.shape[2]),
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate = args.learning_rate,
                loss = args.loss
            )
        elif args.model_type == 'long_term':
            runner = models.RNNLongTermModelRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                number_of_dense_layers=args.number_of_dense_layers,
                dense_layer_size=args.dense_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                input_num_features=int(train_X.shape[2]),
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate = args.learning_rate,
                rnn_to_dense_connection=args.rnn_to_dense_connection,
                loss=args.loss,
                regularization=args.regularization,
                regularization_coefficient=args.regularization_coefficient,
                regularization_application=args.regularization_application,
            )
        elif args.model_type == 'long_term_fusion':
            runner = models.FusionModelRunner(
                node_type=args.layer_type,
                number_of_rnn_layers=args.number_of_rnn_layers,
                rnn_layer_size=args.rnn_layer_size,
                number_of_final_dense_layers=args.number_of_dense_layers,
                number_of_fusion_weather_layers=args.number_of_fusion_weather_layers,
                dense_layer_size=args.dense_layer_size,
                direction=args.direction,
                input_ts_length=loader.run_config['input_ts_length'],
                input_num_recurrent_features=int(len(loader.run_config['recurrent_idxs'])),
                weather_shape=train_X[1].shape,
                output_num_features=len(loader.run_config['y_idxs']),
                normalization_factors=loader.run_config['normalization_factors'],
                y_idxs=loader.run_config['y_idxs'],
                columns=loader.run_config['columns'],
                learning_rate = args.learning_rate,
                rnn_to_dense_connection=args.rnn_to_dense_connection,
                loss=args.loss,
                regularization=args.regularization,
                regularization_coefficient=args.regularization_coefficient,
                regularization_application=args.regularization_application,
                recurrent_idxs=loader.run_config['recurrent_idxs'],
                fusion_layer_structure = args.fusion_layer_structure,
                output_feature_size=args.output_feature_size,
                conv_kernel_size=args.conv_kernel_size,
                conv_stride_size=args.conv_stride_size,
                pool_size=args.pool_size
            )



        # Log the model's architecture
        runner.compile()

        # Set up autologging of performance along the way
        if config.logging:
            mlflow.tensorflow.autolog(log_models=True)

        patience = 3 if args.debug else 30
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks = [early_stopping]

        # Fit model
        load_ts = time.time()
        if config.logging:
            log_param('time_to_load', (load_ts - start_ts) / (60 ** 2))


        train_data = loading.DataGenerator(train_X, train_Y_labels, args.batch_size, shuffle=True)

        valid_data = loading.DataGenerator(valid_X, valid_Y_labels, args.batch_size, shuffle=True)
        kwargs = {'max_queue_size':len(train_data)//2}

        if args.model_type == 'attention_seq2seq':
            # valid_data = (valid_X, valid_Y_labels)

            td = train_data
            vd = valid_data
            def wrapped_train_generator():
                for idx in range(len(td)):
                    if idx == len(td) - 1:
                        data = td[idx]
                        td.on_epoch_end()
                    else:
                        data = td[idx]
                    yield data

            x_shape = list(train_data[0][0].shape)
            y_shape = list(train_data[0][1].shape)
            steps_per_epoch = len(train_data)
            train_data = tf.data.Dataset.from_generator(
                wrapped_train_generator, (tf.float32, tf.float32),
                output_shapes=(tf.TensorShape(x_shape), tf.TensorShape(y_shape))).repeat().prefetch(steps_per_epoch//4)
            kwargs = {'steps_per_epoch':steps_per_epoch}

        del train_X, train_Y_labels
        del valid_X, valid_Y_labels

        model_history = runner.fit(train_data,
                                   epochs=1000,
                                   batch_size=args.batch_size, # This is a bit precarious, as tensorflow says not to
                                   # include a batch size kwarg if your data is a tf.keras.utils.Sequence. I've
                                   # decided to do so anyway, because A) not doing so causes an error with MLFLOW's
                                   # autologging (it tries to log the batch size as 'None', and experiences a silent error
                                   # then quits, because it already logged the batch size as whatever the actual value
                                   # is. This causes it to not save the model in the end), and B) even though TF
                                   # says not to specify batch size, I dug into their code a
                                   # bit and doing so does not seem to have any effect on the model training.
                                   validation_data=valid_data,
                                   verbose=2, callbacks=callbacks,
                                   shuffle=True,
                                   **kwargs)

        train_ts = time.time()
        if config.logging:
            log_param('time_to_train', (train_ts - load_ts) / (60 ** 2))
        valid_X_long_term = loader.load_set('valid', 'test', 'x')
        valid_Y_long_term = loader.load_set('valid', 'test', 'y')

        valid_predictions, valid_errors, hour_haversine_distances_validation = runner.predict(valid_X_long_term, valid_Y_long_term, args)
        predict_ts = time.time()
        save_predictions_and_errors(valid_predictions, valid_errors, args, 'validation')


        test_X_long_term = loader.load_set('test', 'test', 'x')
        test_Y_long_term = loader.load_set('test', 'test', 'y')

        test_predictions, test_errors, hour_haversine_distances_test = runner.predict(test_X_long_term, test_Y_long_term, args)
        save_predictions_and_errors(test_predictions, test_errors, args, 'test')



        #################################
        manager.close()

    if config.logging:
        log_param('time_to_predict', (predict_ts - train_ts) / (60 ** 2))
        log_param('time_in_total', (predict_ts - start_ts) / (60 ** 2))
        if args.model_type in ['iterative','attention_seq2seq']:
            for i in range(3):
                log_metric(f'haversine_validation_loss_{i+1}_hr', float(hour_haversine_distances_validation[i]))
                log_metric(f'haversine_validation_loss_{i+1}_hr_REMEASURED', float(hour_haversine_distances_validation[i]))
                log_metric(f'haversine_test_loss_{i+1}_hr', float(hour_haversine_distances_test[i]))
                log_metric(f'haversine_test_loss_{i+1}_hr_REMEASURED', float(hour_haversine_distances_test[i]))
        elif args.model_type in ['long_term', 'long_term_fusion']:
            log_metric(f'haversine_validation_loss_{args.hours_out}_hr', float(hour_haversine_distances_validation[0]))
            log_metric(f'haversine_validation_loss_{args.hours_out}_hr_REMEASURED', float(hour_haversine_distances_validation[0]))
            log_metric(f'haversine_test_loss_{args.hours_out}_hr', float(hour_haversine_distances_test[0]))
            log_metric(f'haversine_test_loss_{args.hours_out}_hr_REMEASURED', float(hour_haversine_distances_test[0]))

        if args.model_type == 'attention_seq2seq':
            runner.save('model.h5')
            log_artifact('model.h5')
        elif args.model_type in ['iterative','long_term','long_term_fusion']:
            pass

