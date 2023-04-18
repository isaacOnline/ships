import numpy as np
import tensorflow as tf
from haversine import haversine_vector, Unit
from tensorflow.keras.layers import GRU as GRUTF, LSTM as LSTMTF
from tensorflow.keras.optimizers import Adam as AdamTF

from loading.loading import _find_current_col_idx
from loading import Normalizer
from models.seq2seq_model_pieces import TrainTranslator
from models.model_runner import ModelRunner
from models.losses import HaversineLoss


class Seq2SeqRNNAttentionRunner(ModelRunner):
    """
    Class for creating the desired type of Tensorflow model. Creates the model object, and also provides a wrapper
    function for making predictions
    """
    def __init__(self, node_type, number_of_rnn_layers, rnn_layer_size, direction, input_ts_length, output_ts_length,
                 input_num_features, output_num_features, normalization_factors, y_idxs, columns, learning_rate, loss):
        if node_type.lower() == 'gru':
            self.rnn_layer = GRUTF
        elif node_type.lower() == 'lstm':
            self.rnn_layer = LSTMTF
        else:
            raise ValueError('node_type must either be "gru" or "lstm"')

        if direction in ['forward_only', 'bidirectional']:
            self.direction = direction
        else:
            raise ValueError('direction must be either "forward_only" or "bidirectional"')

        self.number_of_rnn_layers = number_of_rnn_layers

        self.rnn_layer_size = rnn_layer_size
        self.input_ts_length = input_ts_length
        self.output_ts_length = output_ts_length
        self.input_num_features = input_num_features
        self.output_num_features = output_num_features

        self._init_model()
        self.normalization_factors = normalization_factors
        self.y_idxs = y_idxs
        self.columns = columns
        self.optimizer = AdamTF(learning_rate=learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError() if loss == 'mse' else HaversineLoss(normalization_factors).haversine_loss



    def _init_model(self):
        """
        Create model as specified during initialization

        :return:
        """
        self.model = TrainTranslator(units = self.rnn_layer_size,
                                     num_input_variables=self.input_num_features,
                                     num_output_variables=self.output_num_features,
                                     input_series_length=self.input_ts_length,
                                     output_series_length=self.output_ts_length)

    def predict(self, input_text, output_text, args):
        """
        Make predictions for an evaluation dataset, returning both the predictions and errors

        Because of a constraint by the model object,the data has to be predicted in batches. This is handled within the
        predict method

        :param input_text: Dataset to make predictions for
        :param output_text: Ground truth
        :param args: argparse.Namespace specifying model
        :return:
        """
        input_text = input_text.astype(np.float32)
        chunks = np.split(input_text,np.arange(args.batch_size,input_text.shape[0], args.batch_size))
        preds = [self.model.predict(t).numpy() for t in chunks[:-1]]
        preds += [self.model.predict(input_text[-args.batch_size:]).numpy()[-chunks[-1].shape[0]:]]# get last chunk (which is not correct size)
        result_tokens = np.concatenate(preds)
        predicted_lat_long = Normalizer().unnormalize(result_tokens, self.normalization_factors)

        output_text = Normalizer().unnormalize(output_text, self.normalization_factors)
        lat_lon_idxs = [_find_current_col_idx(c, self.columns) for c in ['lat','lon']]
        mean_haversine_distances = []
        haversine_distances = []
        for i in range(self.output_ts_length):
            haversine_distance = haversine_vector(output_text[:, i, lat_lon_idxs],
                                                  predicted_lat_long[:, i, lat_lon_idxs],
                                                  Unit.KILOMETERS)
            haversine_distances.append(haversine_distance)
            mean_haversine_distance = haversine_distance.mean()
            mean_haversine_distances.append(mean_haversine_distance)

        common_prediction_time = 60
        hour_idxs = [int((h - common_prediction_time) / args.time) - 1 for h in [120, 180, 240]]

        Y_hats = [predicted_lat_long[:, i, lat_lon_idxs] for i in hour_idxs]
        hour_haversine_distances = [haversine_distances[i] for i in hour_idxs]
        mean_hour_haversine_distances = [mean_haversine_distances[i] for i in hour_idxs]
        return Y_hats, hour_haversine_distances, mean_hour_haversine_distances


    def save(self, path):
        """
        Save model to disk

        :param pos_args: Positional args, passed down to model object's save method
        :param named_args: Named args, passed down to model object's save method
        :return:
        """
        tf.saved_model.save(self.model, path)