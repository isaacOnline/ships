import numpy as np
import pandas as pd
from haversine import haversine_vector, Unit
from keras import Input, Model
from keras.layers import GRU as GRUKeras, LSTM as LSTMKeras, Bidirectional, Dense, Flatten
from keras.optimizer_v2.adam import Adam as AdamKeras

from loading import Normalizer
from loading.loading import _find_current_col_idx
from models.model_runner import ModelRunner
from models.losses import HaversineLoss

class RNNModelRunner(ModelRunner):
    """
    Class for creating the desired type of Tensorflow model. Creates the model object, and also provides a wrapper
    function for making predictions
    """
    def __init__(self, node_type, number_of_rnn_layers, rnn_layer_size, number_of_dense_layers, dense_layer_size,
                 direction, input_ts_length, input_num_features, output_num_features, normalization_factors,
                 y_idxs, columns, learning_rate, rnn_to_dense_connection, loss='mse'):
        if node_type.lower() == 'gru':
            self.rnn_layer = GRUKeras
        elif node_type.lower() == 'lstm':
            self.rnn_layer = LSTMKeras
        else:
            raise ValueError('node_type must either be "gru" or "lstm"')

        if direction in ['forward_only','bidirectional']:
            self.direction = direction
        else:
            raise ValueError('direction must be either "forward_only" or "bidirectional"')

        if not np.all([i in range(len(y_idxs)) for i in y_idxs]):
            raise ValueError('Y indexes must be the first columns in the dataset in order for normalization to work '
                             'properly')


        self.number_of_rnn_layers = number_of_rnn_layers
        self.rnn_layer_size = rnn_layer_size
        self.number_of_dense_layers = number_of_dense_layers
        self.dense_layer_size = dense_layer_size
        self.ts_length = input_ts_length
        self.input_num_features = input_num_features
        self.output_size = output_num_features
        self.rnn_to_dense_connection = rnn_to_dense_connection
        self._init_model()
        self.normalization_factors = normalization_factors
        self.y_idxs = y_idxs
        self.columns = columns
        self.optimizer = AdamKeras(learning_rate=learning_rate)
        self.loss = 'mse' if loss=='mse' else HaversineLoss(normalization_factors).haversine_loss


    def _init_model(self):
        """
        Create model as specified during initialization

        :return:
        """
        input = Input(shape=(self.ts_length, self.input_num_features))

        hidden = [input]
        if self.rnn_to_dense_connection == 'all_nodes':
            num_full_sequence_layers = self.number_of_rnn_layers
        else:
            num_full_sequence_layers = self.number_of_rnn_layers - 1

        if num_full_sequence_layers > 0:
            if self.direction == 'forward_only':
                hidden.append(self.rnn_layer(self.rnn_layer_size, return_sequences=True)(hidden[-1]))
                num_full_sequence_layers -= 1
            elif self.direction == 'bidirectional':
                hidden.append(Bidirectional(self.rnn_layer(self.rnn_layer_size, return_sequences=True))(hidden[-1]))
                num_full_sequence_layers -= 1

        for layer in range(num_full_sequence_layers):
            if self.direction == 'forward_only':
                hidden.append(self.rnn_layer(self.rnn_layer_size, return_sequences=True)(hidden[-1]))
            elif self.direction == 'bidirectional':
                hidden.append(Bidirectional(self.rnn_layer(self.rnn_layer_size, return_sequences=True))(hidden[-1]))

        if self.rnn_to_dense_connection == 'all_nodes':
            hidden.append(Flatten('channels_first')(hidden[-1]))
        else:
            if self.direction == 'forward_only':
                hidden.append(self.rnn_layer(self.rnn_layer_size, return_sequences=False)(hidden[-1]))
            elif self.direction == 'bidirectional':
                hidden.append(Bidirectional(self.rnn_layer(self.rnn_layer_size, return_sequences=False))(hidden[-1]))



        # Define Dense section
        for layer in range(self.number_of_dense_layers):
            hidden.append(Dense(self.dense_layer_size, activation='relu')(hidden[-1]))

        # Define output
        output = Dense(self.output_size, activation='linear')(hidden[-1])

        self.model = Model(inputs=input, outputs=output)


    def save(self, *pos_args, **named_args):
        """
        Save model to disk

        :param pos_args: Positional args, passed down to model object's save method
        :param named_args: Named args, passed down to model object's save method
        :return:
        """
        self.model.save(*pos_args, **named_args)

    def insert_predictions(self, X, predictions, time):
        """
        Insert predicted values into X dataset

        The iterative model makes predictions for the next timestamp, then uses these as input data
        to predict the following timestamp.

        This method is used for amending the input dataset so that new predictions can be made. It appends a set of
        predicted values to the end of the previous input data, and cuts off that input data's first timestamp

        :param X: Original input data
        :param predictions: Predictions to append
        :param time: The time gap being used
        :return:
        """
        # Copy over the static info
        Y_hat_i_full = Normalizer().unnormalize(X[:, -1,:].copy(), self.normalization_factors)

        predictions = Normalizer().unnormalize(predictions.copy(), self.normalization_factors)

        # Input the new predictions
        Y_hat_i_full[:, self.y_idxs] = predictions

        # Distance traveled
        if 'distance_traveled' in self.columns['column'].values:
            lat_lon_idx = [_find_current_col_idx(c,self.columns) for c in ['lat','lon']]
            first_lat_lon =  Normalizer().unnormalize(X[:, 1, lat_lon_idx], self.normalization_factors).copy()
            predicted_lat_lon = Y_hat_i_full[:,lat_lon_idx]
            distance_traveled = haversine_vector(first_lat_lon, predicted_lat_lon, Unit.KILOMETERS)
            dt_idx = _find_current_col_idx('distance_traveled',self.columns)
            Y_hat_i_full[:, dt_idx] = distance_traveled

        # Hour/day
        if 'day_of_week' in self.columns['column'].values:
            hour_col = _find_current_col_idx('hour',self.columns)
            dow_col = _find_current_col_idx('day_of_week',self.columns)
            incremented = (Normalizer().unnormalize(X, self.normalization_factors)[:,1:, hour_col]
                           - Normalizer().unnormalize(X, self.normalization_factors)[:,:-1, hour_col]) != 0
            incremented = pd.DataFrame({'r': np.where(incremented)[0], 'c': np.where(incremented)[1]})
            last_incremented = incremented.groupby('r').max().to_numpy().squeeze()

            how_often_to_increment = 60 / time

            Y_hat_i_full[:, hour_col] = np.where(
                last_incremented + how_often_to_increment < X.shape[1],
                Y_hat_i_full[:, hour_col] + 1,
                Y_hat_i_full[:, hour_col]
            )
            Y_hat_i_full[:, dow_col] = np.where(
                Y_hat_i_full[:, hour_col] > 23,
                Y_hat_i_full[:, dow_col] + 1,
                Y_hat_i_full[:, dow_col]
            )
            Y_hat_i_full[:, hour_col] %= 24
            Y_hat_i_full[:, dow_col] %= 7

        # Drop off the first timestamp
        layers = [X[:, j, :] for j in range(1, X.shape[1])]

        # Add in the prediction from last round as the last timestamp
        Y_hat_i_full = Normalizer().normalize_data(Y_hat_i_full, self.normalization_factors)
        layers.append(Y_hat_i_full)

        # Stack everything together and append to the list
        new_X = np.stack(layers, axis=1)
        return new_X

    def predict(self, valid_X_long_term, valid_Y_long_term, args):
        """
        Make predictions for an evaluation dataset, returning both the predictions and errors

        :param valid_X_long_term: Dataset to make predictions for
        :param valid_Y_long_term: Ground truth
        :param args: argparse.Namespace specifying model
        :return:
        """
        valid_Xs = [valid_X_long_term]
        valid_Y_hats = []
        mean_haversine_distances = []
        haversine_distances = []
        # Iteratively carry predictions forward
        for i in range(valid_Y_long_term.shape[1]):
            # Valid Xs is a list containing the datasets used for prediction. It is appended to as we make new predictions
            # and use those for prediction
            X = valid_Xs[i]

            # Make prediction
            Y_hat_i_normalized = self.model.predict(X)
            Y_hat_i_unnormalized = Normalizer().unnormalize(Y_hat_i_normalized, self.normalization_factors)
            valid_Y_hats.append(Y_hat_i_unnormalized)


            # Get haversine distance
            lat_lon_idxs = [_find_current_col_idx(c, self.columns) for c in ['lat','lon']]
            ground_truth_Y_unnormalized = Normalizer().unnormalize(valid_Y_long_term[:, i, lat_lon_idxs], self.normalization_factors)

            haversine_distance = haversine_vector(ground_truth_Y_unnormalized,
                                                  Y_hat_i_unnormalized[:, lat_lon_idxs],
                                                  Unit.KILOMETERS)
            haversine_distances.append(haversine_distance)
            mean_haversine_distance = haversine_distance.mean()
            mean_haversine_distances.append(mean_haversine_distance)

            # If this isn't the last prediction we needed to make, append to the list of X sets
            if i != valid_Y_long_term.shape[1] - 1:
                valid_Xs.append(self.insert_predictions(X, Y_hat_i_normalized, args.time))

        # TODO: Make more adjustable
        common_prediction_time = 60
        hour_idxs = [int((h - common_prediction_time) / args.time) - 1 for h in [120, 180, 240]]

        valid_Y_hats = [valid_Y_hats[i] for i in hour_idxs]
        hour_haversine_distances = [haversine_distances[i] for i in hour_idxs]
        mean_hour_haversine_distances = np.array(mean_haversine_distances)[hour_idxs]
        return valid_Y_hats, hour_haversine_distances, mean_hour_haversine_distances