from math import pi

import tensorflow as tf
from config import config


class HaversineLoss():
    """
    Class implementing haversine loss
    """
    def __init__(self, normalization_factors):
        self.lat_min = normalization_factors['lat']['min']
        self.lat_range = normalization_factors['lat']['max'] - self.lat_min

        self.lon_min = normalization_factors['lon']['min']
        self.lon_range = normalization_factors['lon']['max'] - self.lon_min

    @tf.function
    def haversine_loss(self, y_true, y_pred):
        """
        Calculate haversine distance between two sets of points

        Coordinates should be in degrees, scaled to be between 0 and 1 based on the lat/lon mins in the config file

        Latitudes should be in the 0th column in both sets, and longitudes should be in the first columns

        :param y_true: Ground truth points
        :param y_pred: Predicted points
        :return:
        """
        # If this is just being traced, the batch_size should be None, which
        y_true = tf.cast(y_true, y_pred.dtype)
        if len(y_true.shape) == 2:
            batch_axis = 0
            lat_lon_axis = 1
            num_predictions = tf.shape(y_true)[batch_axis]
        elif len(y_true.shape) == 3:
            batch_axis = 0
            time_axis = 1
            lat_lon_axis = 2
            num_predictions = tf.shape(y_true)[batch_axis] * tf.shape(y_true)[time_axis]
        else:
            raise ValueError('Unknown input shape')


        lat_min = config.dataset_config.lat_1
        lat_range = config.dataset_config.lat_2 - lat_min

        lon_min = config.dataset_config.lon_1
        lon_range = config.dataset_config.lon_2 - lon_min

        y_true_deg = (y_true * tf.constant([lat_range, lon_range], dtype=y_true.dtype)
                      + tf.constant([lat_min, lon_min],dtype=y_true.dtype)) * tf.constant(pi / 180., dtype=y_true.dtype)
        lat_true = tf.gather(y_true_deg, 0, axis=lat_lon_axis)
        lon_true = tf.gather(y_true_deg, 1, axis=lat_lon_axis)

        y_pred_deg = (y_pred * tf.constant([lat_range, lon_range], dtype = y_pred.dtype)
                      + tf.constant([lat_min, lon_min], dtype=y_pred.dtype)) * tf.constant(pi / 180., dtype=y_pred.dtype)
        lat_pred = tf.gather(y_pred_deg, 0, axis=lat_lon_axis)
        lon_pred = tf.gather(y_pred_deg, 1, axis=lat_lon_axis)

        EARTH_RADIUS = 6371

        interior = (tf.math.sin((lat_true - lat_pred)/2) ** 2
                   + tf.math.cos(lat_true) * tf.math.cos(lat_pred) * (tf.math.sin((lon_true - lon_pred)/2) ** 2))

        # Clip, to make sure there aren't any floating point issues
        interior = tf.clip_by_value(interior, 0., 1.)

        interior = tf.math.sqrt(interior)

        distance = 2 * EARTH_RADIUS * tf.math.asin(
            interior
        )

        # Only keep places where distance is not 0, as due to the sqrt above, these will not be differentiable and will
        # result in a NaN. These are still still ~kept in~ the loss by the fact that we are averaging using the batch
        # size below, (so they still contribute to the mean), it's just that we aren't using them for optimization,
        # which is fine as they are already perfect.
        distance = tf.reshape(distance,[-1])
        distance = tf.squeeze(tf.gather(distance, tf.where(distance != 0)))

        mean_distance = tf.reduce_sum(distance) / tf.cast(num_predictions, distance.dtype)

        return mean_distance


    def get_config(self):
        """
        Return the lat/lon min/max, which can be used for saving a loss object to disk

        :return:
        """
        return {'lat_min': self.lat_min,
                'lat_range': self.lat_range,
                'lon_min': self.lon_min,
                'lon_range': self.lon_range}

    def from_config(self, config):
        """
        Save the lat/lon to this object, based on a config loaded from somewhere else

        :param config:
        :return:
        """
        for k, v in config.items():
            setattr(self, k, v)