from haversine import haversine_vector, Unit
from keras import Input, Model
from keras.layers import GRU as GRUKeras, LSTM as LSTMKeras, Bidirectional, Flatten, Dense, Dropout
from keras.optimizer_v2.adam import Adam as AdamKeras
from keras.regularizers import L1, L2

from loading import add_distance_traveled, Normalizer
from models.losses import HaversineLoss
from models.model_runner import ModelRunner


class RNNLongTermModelRunner(ModelRunner):
    """
    Class for creating the desired type of Tensorflow model. Creates the model object, and also provides a wrapper
    function for making predictions
    """
    def __init__(self, node_type, number_of_rnn_layers, rnn_layer_size, number_of_dense_layers, dense_layer_size,
                 direction, input_ts_length, input_num_features, output_num_features, normalization_factors,
                 y_idxs, columns, learning_rate, rnn_to_dense_connection, loss='mse',
                 regularization = None, regularization_application=None, regularization_coefficient=None):
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

        self.number_of_rnn_layers = number_of_rnn_layers
        self.rnn_layer_size = rnn_layer_size
        self.number_of_dense_layers = number_of_dense_layers
        self.dense_layer_size = dense_layer_size
        self.ts_length = input_ts_length
        self.input_num_features = input_num_features
        self.output_size = output_num_features
        self.rnn_to_dense_connection = rnn_to_dense_connection
        if regularization == 'dropout':
            if regularization_application == 'recurrent':
                self.rnn_regularization = {'recurrent_dropout':regularization_coefficient}
                self.dense_dropout = 0.0
                self.dense_regularization = {}
            elif regularization_application is None:
                self.rnn_regularization = {'dropout':regularization_coefficient}
                self.dense_dropout = regularization_coefficient
                self.dense_regularization = {}
        elif regularization in ['l1','l2']:
            self.regularizer = L1 if regularization == 'l1' else L2
            if regularization_application in ['bias','activity']:
                self.rnn_regularization = {f'{regularization_application}_regularizer':
                                                 self.regularizer(regularization_coefficient)}
                self.dense_dropout = 0.0
                self.dense_regularization = {f'{regularization_application}_regularizer':
                                                 self.regularizer(regularization_coefficient)}
            elif regularization_application == 'recurrent':
                self.rnn_regularization = {f'{regularization_application}_regularizer':
                                                 self.regularizer(regularization_coefficient)}
                self.dense_dropout = 0.0
                self.dense_regularization = {}
        else:
            self.rnn_regularization = {}
            self.dense_dropout = 0.0
            self.dense_regularization = {}


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
        recurrent_input = Input(shape=(self.ts_length, self.input_num_features))

        if self.rnn_to_dense_connection == 'all_nodes':
            num_full_sequence_layers = self.number_of_rnn_layers
        else:
            num_full_sequence_layers = self.number_of_rnn_layers - 1

        # Define RNN section
        if self.direction == 'forward_only':
            hidden = [self.rnn_layer(self.rnn_layer_size, return_sequences=True,
                                     **self.rnn_regularization)(recurrent_input)]
            num_full_sequence_layers -= 1
        elif self.direction == 'bidirectional':
            hidden = [Bidirectional(self.rnn_layer(self.rnn_layer_size, return_sequences=True,
                                                   **self.rnn_regularization))(recurrent_input)]
            num_full_sequence_layers -= 1

        for layer in range(num_full_sequence_layers):
            if self.direction == 'forward_only':
                hidden.append(self.rnn_layer(self.rnn_layer_size, return_sequences=True,
                                             **self.rnn_regularization)(hidden[-1]))
            elif self.direction == 'bidirectional':
                hidden.append(Bidirectional(self.rnn_layer(self.rnn_layer_size, return_sequences=True,
                                                           **self.rnn_regularization))(hidden[-1]))

        if self.rnn_to_dense_connection == 'all_nodes':
            hidden.append(Flatten('channels_first')(hidden[-1]))
        else:
            if self.direction == 'forward_only':
                hidden.append(self.rnn_layer(self.rnn_layer_size, return_sequences=False,
                                             **self.rnn_regularization)(hidden[-1]))
            elif self.direction == 'bidirectional':
                hidden.append(Bidirectional(self.rnn_layer(self.rnn_layer_size, return_sequences=False,
                                                           **self.rnn_regularization))(hidden[-1]))

        # Define Dense section
        for layer in range(self.number_of_dense_layers):
            hidden.append(Dense(self.dense_layer_size, activation='relu',
                                **self.dense_regularization)(hidden[-1]))
            hidden.append(Dropout(self.dense_dropout)(hidden[-1]))

        # Define output
        output = Dense(self.output_size, activation='linear')(hidden[-1])

        self.model = Model(inputs=recurrent_input, outputs=output)

    def save(self, *pos_args, **named_args):
        """
        Save model to disk

        :param pos_args: Positional args, passed down to model object's save method
        :param named_args: Named args, passed down to model object's save method
        :return:
        """
        self.model.save(*pos_args, **named_args)

    def predict(self, valid_X_long_term, valid_Y_long_term, args):
        """
        Make predictions for an evaluation dataset, returning both the predictions and errors

        :param valid_X_long_term: Dataset to make predictions for
        :param valid_Y_long_term: Ground truth
        :param args: argparse.Namespace specifying model
        :return:
        """
        Y_hat = self.model.predict(valid_X_long_term)
        Y_hat = Normalizer().unnormalize(Y_hat, self.normalization_factors)
        valid_Y_long_term = Normalizer().unnormalize(valid_Y_long_term, self.normalization_factors)

        haversine_distances = haversine_vector(valid_Y_long_term, Y_hat, Unit.KILOMETERS)
        mean_haversine_distance = haversine_distances.mean()
        return [Y_hat], [haversine_distances], [mean_haversine_distance]