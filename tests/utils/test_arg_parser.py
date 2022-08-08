import argparse
import os

import numpy as np

from config import config
from config.dataset_config import datasets
from utils.arg_validation import NotGiven, Given, Values, ValueRange, Req


class TestArgParser():
    """
    Class for parsing arguments for test script
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.transformer_only_args = ['number_of_heads', 'number_of_transformer_layers',
                                      'number_of_lat_lon_embeddings', 'number_of_sog_cog_embeddings',
                                      'number_of_year_embeddings','number_of_month_embeddings',
                                      'month_year','vessel_type', 'final_tokens','warmup_tokens',
                                      'lat_lon_bin_multiplier', 'n_sog_bins','n_cog_bins','weather_embeddings_per_var']
        self.long_term_only_args = ['hours_out','regularization',
                                    'regularization_application','regularization_coefficient']
        self.fusion_only_args = ['number_of_fusion_weather_layers']
        self.convolution_only_args = ['output_feature_size', 'conv_kernel_size', 'conv_stride_size', 'pool_size']
        self._add_args()

    def _add_args(self):
        """
        Add relevant arguments

        :return: None
        """
        # Main params
        self.parser.add_argument('time', type=int, help='Time gap to consider')
        self.parser.add_argument('dataset_name', type=str, choices=datasets.keys(), help='Data set name to use')
        self.parser.add_argument('--model_type',type=str,default='iterative',
            choices=['iterative','attention_seq2seq', 'long_term', 'long_term_fusion'],help='transformer model_type is not available in this repository')
        self.parser.add_argument('-s', '--seed', type=str, default='None',
            help='Random seed to use when creating validation set')
        self.parser.add_argument('--hours_out', type=int, choices=[1, 2, 3],
            help='If doing long term prediction, the number of hours to predict into the future')

        # Tools for debugging
        self.parser.add_argument('-nl', '--no_logging', action='store_true',
                            help='If included, will not log run to MLflow')
        self.parser.add_argument('-d', '--debug', action='store_true',
                            help='If true, will only use a sample of data when training')

        # Data preprocessing
        self.parser.add_argument('--time_of_day', type=str, choices=['ignore', 'hour_day'], default='ignore', help='Whether or not to include hour of day/day of week as features.')
        self.parser.add_argument('--sog_cog', type=str, default='raw',choices=['ignore','raw','min_median_max','median'], help='The method for including sog/cog. "raw" will include the interpolated sog/cog values at a given timestamp. "min_median_max", and "median" will summarize sog/cog instead of including their raw values')
        self.parser.add_argument('--weather', type=str,default='ignore',choices=['ignore','currents'],help='Whether or not weather data should be used')
        self.parser.add_argument('--weather_processing',choices=['ignore','embedded','raw'],help='Will be set appropriately if not passed in. (Should be one of "ignore" or "raw", if model_type is not "transformer", and should match weather arg. This will be selected for you if you do not pass in a value.)')
        self.parser.add_argument('--extended_recurrent_idxs',type=str,default=None, choices=[None,'vt_dst_and_time', 'all_non_weather'],help='If using a fusion model, whether the recurrent input should include vessel type, destination and time, or all non weather variables')
        self.parser.add_argument('--destination', type=str,default='ignore',choices=['ignore','cluster_centers','ohe'],help="How to use destination information - one hot encoded, or with columns specifying where the destination clusters' centers are")
        self.parser.add_argument('--length_of_history',type=int,default=3, choices=[1,2,3], help='How many hours of history to use')

        # NN Architecture
        self.parser.add_argument('--layer_type', type=str, default=None, choices=['lstm', 'gru', 'sample', None])
        self.parser.add_argument('--direction', type=str, default='sample', choices=['forward_only', 'bidirectional', 'sample'])
        self.parser.add_argument('--number_of_dense_layers', type=int)
        self.parser.add_argument('--dense_layer_size', type=int)
        self.parser.add_argument('--number_of_rnn_layers', type=int)
        self.parser.add_argument('--rnn_layer_size', type=int)
        self.parser.add_argument('--number_of_fusion_weather_layers', type=int)
        self.parser.add_argument('--rnn_to_dense_connection', type=str, default=None,
                                 choices=['all_nodes','final_node',None],
                                 help='If doing long term prediction, how to connect the rnn layers to the dense '
                                      'layers. If "all_nodes" is specified, the entire rnn sequence will be fed into '
                                      'the dense layer, otherwise if "final_node" is specified, only the output of the '
                                      'final rnn node will go into the dense layer')

        # NN Learning
        self.parser.add_argument('--loss', type=str, default='mse',choices=['mse','haversine'])
        self.parser.add_argument('--batch_size', type=int)
        self.parser.add_argument('--learning_rate', type=float)



        self.parser.add_argument('--month_year', type=str, choices=['ignore', 'embedded', 'raw', None], help='Method for including month/year data in a transformer model. For other models, the month/year is included by default and cannot be removed. Do not specify if model_type is not "transformer".')
        self.parser.add_argument('--vessel_type', type=str, choices=['ignore', 'ohe', None], help='Method for including vessel type data in a transformer model. For other models, the vessel type is included by default and cannot be removed. Do not specify if model_type is not "transformer".')
        # The fusion_layer_structure option was originally included to specify if you wanted to process the weather
        # data using convolutional layers. This option has been deprecated, as we did not find that it improved
        # performance, so the 'convolutions' value is not longer an option
        self.parser.add_argument('--fusion_layer_structure',choices=[None,'dense'], help='Do not specify, as default will be chosen appropriately for you.')
        self.parser.add_argument('--n_sog_bins', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--n_cog_bins', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--weather_bins_per_var', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--number_of_heads', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--number_of_transformer_layers', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--number_of_lat_lon_embeddings', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--warmup_tokens', type=float, help='Only specify for transformer models')
        self.parser.add_argument('--final_tokens', type=float, help='Only specify for transformer models')
        self.parser.add_argument('--number_of_sog_cog_embeddings', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--number_of_year_embeddings', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--number_of_month_embeddings', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--weather_embeddings_per_var', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--lat_lon_bin_multiplier', type=int, help='Only specify for transformer models')
        self.parser.add_argument('--conv_kernel_size', type = int, default=None,help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--conv_stride_size', type = int, default=None,help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--pool_size', type = int, default=None,help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--regularization',type=str,choices=['dropout','l1','l2',None],help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--regularization_application',type=str,choices=['recurrent','bias','activity',None],help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--regularization_coefficient',type=float,help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--median_stopping',type=str,
                                 default='do_not_use',
                                 choices=['do_not_use'],
                                 help = 'DEPRECATED')
        self.parser.add_argument('--output_feature_size', type = int, default=None,help='DEPRECATED. DO NOT SPECIFY.')
        self.parser.add_argument('--distance_traveled', type=str, default='ignore', choices=['ignore'], help='DEPRECATED')



    def parse(self):
        """
        Parse arguments

        :return: argparse.Namespace
        """
        self.args = self.parser.parse_args()
        config.dataset_config = datasets[self.args.dataset_name]
        config.box_and_year_dir = os.path.join(
            config.data_directory,
            f'{config.dataset_config.lat_1}_{config.dataset_config.lat_2}_'
            f'{config.dataset_config.lon_1}_{config.dataset_config.lon_2}_'
            f'{config.start_year}_{config.end_year}')

        if self.args.no_logging:
            config.logging = False

        self._sample_args()
        self._validate_args()

        return self.args

    def _sample_args(self):
        """
        If any arguments, e.g. features of the NN architecture, need to be randomly sampled, do so

        :return:
        """
        # Sample batch size if one is not specified (must be between 128 and 4096, and a power of 2)
        if self.args.batch_size is None:
            self.args.batch_size = np.random.choice([2 ** x for x in [7, 8, 9, 10, 11, 12]])
        # Sample learning rate if one is not given
        if self.args.learning_rate is None:
            self.args.learning_rate = np.exp(np.random.uniform(1, -14))

        if self.args.weather_processing is None:
            if self.args.weather == 'ignore':
                self.args.weather_processing = 'ignore'
            else:
                if self.args.model_type == 'transformer':
                    self.args.weather_processing = 'embedded'
                else:
                    self.args.weather_processing = 'raw'

        if self.args.fusion_layer_structure is None:
            if self.args.model_type == 'long_term_fusion':
                if self.args.weather == 'currents':
                    self.args.fusion_layer_structure = 'dense'
                elif self.args.extended_recurrent_idxs == 'all_non_weather':
                    pass
                else:
                    self.args.fusion_layer_structure = 'dense'
            else:
                pass

        if self.args.model_type in ['iterative','long_term','attention_seq2seq','long_term_fusion']:
            # Sample layer type if one is not specified
            if self.args.layer_type == 'sample':
                self.args.layer_type = np.random.choice(['gru', 'lstm'])
            # Sample direction if one is not specified
            if self.args.direction == 'sample':
                self.args.direction = np.random.choice(['forward_only', 'bidirectional'])
            # Sample number of rnn layers if not given
            if self.args.number_of_rnn_layers is None:
                self.args.number_of_rnn_layers = np.random.randint(1, 6)
            # Sample cell size if not given
            if self.args.rnn_layer_size is None:
                self.args.rnn_layer_size = np.random.randint(50, 351)
            # Sample number of rnn layers if not given
            if self.args.number_of_dense_layers is None and self.args.model_type in ['iterative', 'long_term',
                                                                                     'long_term_fusion']:
                self.args.number_of_dense_layers = np.random.randint(0, 3)
            # Sample cell size if not given
            if self.args.dense_layer_size is None and self.args.model_type in ['iterative', 'long_term',
                                                                               'long_term_fusion']:
                self.args.dense_layer_size = np.random.randint(50, 351)
            if self.args.number_of_fusion_weather_layers is None and self.args.model_type == 'long_term_fusion':
                self.args.number_of_fusion_weather_layers = np.random.randint(0, 5)

            if self.args.fusion_layer_structure == 'convolutions':
                if self.args.output_feature_size is None:
                    self.args.output_feature_size = 16
                if self.args.conv_kernel_size is None:
                    self.args.conv_kernel_size = 3
                if self.args.conv_stride_size is None:
                    self.args.conv_stride_size = 1
                if self.args.pool_size is None:
                    self.args.pool_size = 2

        elif self.args.model_type == 'transformer':

            if self.args.number_of_transformer_layers is None:
                self.args.number_of_transformer_layers = np.random.randint(10)

            if self.args.number_of_lat_lon_embeddings is None:
                self.args.number_of_lat_lon_embeddings = 2 ** np.random.randint(6, 11)

            if self.args.number_of_sog_cog_embeddings is None:
                self.args.number_of_sog_cog_embeddings = 2 ** np.random.randint(4, 10)

            if self.args.final_tokens is None:  # This default taken from the TrAISformer code: 2 * len(aisdatasets["train"]) * cf.max_seqlen
                self.args.final_tokens = 45711842

            if self.args.warmup_tokens is None:  # This default taken from the TrAISformer code: n_embd / 1.5 * 20
                self.args.warmup_tokens = 10240

            if self.args.lat_lon_bin_multiplier is None:
                self.args.lat_lon_bin_multiplier = 100

            if self.args.n_sog_bins is None:
                self.args.n_sog_bins = 30

            if self.args.n_cog_bins is None:
                self.args.n_cog_bins = 72

            if self.args.vessel_type is None:
                self.args.vessel_type = 'ohe'

            if self.args.month_year is None:
                self.args.month_year = 'embedded'

            if self.args.month_year == 'embedded':
                if self.args.number_of_year_embeddings is None:
                    self.args.number_of_year_embeddings = 2 ** np.random.randint(3, 9) - 2

                if self.args.number_of_month_embeddings is None:
                    self.args.number_of_month_embeddings = 2 ** np.random.randint(3, 10) - 4

            if self.args.weather_processing == 'embedded':
                if self.args.weather_embeddings_per_var is None:
                    self.args.weather_embeddings_per_var = 2 ** np.random.randint(3,6)

                if self.args.weather_bins_per_var is None:
                    self.args.weather_bins_per_var = 20

            if self.args.number_of_heads is None:
                n_embed = (
                    self.args.number_of_year_embeddings
                    + self.args.number_of_month_embeddings
                    + 6 # (number of vessel types)
                    + self.args.number_of_lat_lon_embeddings * 2
                    + self.args.number_of_sog_cog_embeddings * 2
                )
                self.args.number_of_heads = n_embed + 1

                while not np.isclose((n_embed / self.args.number_of_heads ) %1, 0):
                    self.args.number_of_heads = 2 ** np.random.randint(2, 9)

        if not self.args.seed or self.args.seed == 'None':
            self.args.seed = np.random.randint(1e8)
        else:
            self.args.seed = int(self.args.seed)

    def _validate_args(self):
        """
        Make sure that args do not have any conflicts

        First specifies a hard-coded list of requirements, then validates that all requirements have been met

        :return:
        """
        requirements = [
            Req(Given('dataset_name')),
            Req(Given('model_type')),
            Req(Given('time')),
            Req(Given('loss')),

            # The hours_out param is only used by the long term models (The other model_types predict all hours_out values at once)
            Req(a=Values('model_type', ['long_term', 'long_term_fusion']), b=Given('hours_out')),
            Req(a=Values('model_type', ['iterative', 'attention_seq2seq', 'transformer']),
                b=NotGiven(['hours_out', 'regularization',
                             'regularization_application', 'regularization_coefficient'])),

            # I only implemented a forward only, single layer seq2seq model, (similar to the paper from Forti et al.,
            # although using an attention mechanism)
            Req(a=Values('model_type', ['attention_seq2seq']),
                b=Values('direction', ['forward_only'])),
            Req(a=Values('model_type', ['attention_seq2seq']),
                b=Values('number_of_rnn_layers', [1])),

            Req(a=Values('model_type',['iterative','attention_seq2seq','long_term','long_term_fusion']),
                b=Given('layer_type')),
            Req(a=Values('model_type', ['transformer']),
                b=NotGiven('layer_type')),

            # If raw SOG/COG are being used for prediction by the iterative/transformer models, this means that they
            # also need to be predicted (because unlike the other features, they are not stable and cannot just be
            # carried forward). Unfortunately this means that Haversine loss cannot be used.
            Req(a=[Values('model_type', ['iterative', 'transformer']), Values('sog_cog', ['raw'])],
                b=Values('loss', ['mse'])),

            Req(a=[Values('model_type', ['iterative', 'long_term', 'long_term_fusion'])],
                b=Given('rnn_to_dense_connection')),
            Req(a=[Values('model_type', ['transformer', 'attention_seq2seq'])],
                b=NotGiven('rnn_to_dense_connection')),

            # Number of fusion layers only works if using a fusion model
            Req(a=Values('model_type', ['long_term_fusion']),
                b=Given('number_of_fusion_weather_layers')),
            Req(a=Values('model_type', ['iterative', 'transformer', 'attention_seq2seq', 'long_term']),
                b=NotGiven('number_of_fusion_weather_layers')),

            # Some conditions are only required for weather processing models
            Req(a=Given('weather'), b=Given('weather_processing')),
            Req(a=NotGiven('weather'), b=NotGiven('weather_processing')),
            Req(a=Values('weather_processing', ['embedded']),
                b=Given('weather_embeddings_per_var')),
            Req(a=Values('weather_processing', ['ignore', 'raw']),
                b=NotGiven('weather_embeddings_per_var')),

            Req(a=Values('model_type', ['iterative', 'attention_seq2seq', 'long_term', 'long_term_fusion']),
                b=Values('weather_processing', ['ignore', 'raw'])),

            Req(a=Values('model_type', ['iterative', 'attention_seq2seq', 'long_term', 'long_term_fusion']),
                b=NotGiven(['number_of_heads', 'number_of_transformer_layers',
                            'number_of_lat_lon_embeddings', 'number_of_sog_cog_embeddings',
                            'number_of_year_embeddings', 'number_of_month_embeddings',
                            'month_year', 'vessel_type', 'final_tokens', 'warmup_tokens',
                            'lat_lon_bin_multiplier', 'n_sog_bins', 'n_cog_bins'])),

            Req(a=Values('model_type', ['transformer']),
                b=Given(['number_of_heads', 'number_of_transformer_layers', 'number_of_lat_lon_embeddings',
                         'number_of_sog_cog_embeddings', 'final_tokens', 'warmup_tokens',
                         'lat_lon_bin_multiplier', 'n_sog_bins', 'n_cog_bins'])),

            Req(a=Values('month_year', ['embedded']),
                b=Given(['number_of_year_embeddings', 'number_of_month_embeddings'])),
            Req(a=Values('month_year', ['ignore', 'raw', None]),
                b=NotGiven(['number_of_year_embeddings', 'number_of_month_embeddings'])),

            # Some args are required if using a convolutional net
            Req(a=Values('fusion_layer_structure', ['convolutions']),
                b=Given(['output_feature_size', 'conv_kernel_size', 'conv_stride_size', 'pool_size'])),
            Req(a=Values('fusion_layer_structure', ['dense', None]),
                b=NotGiven(['output_feature_size', 'conv_kernel_size', 'conv_stride_size', 'pool_size'])),

            # Certain regularization parameters only work in unison
            Req(a=Given('regularization'), b=Given(['regularization_coefficient', 'regularization_application'])),
            Req(a=NotGiven('regularization'), b=NotGiven(['regularization_coefficient', 'regularization_application'])),

            Req(a=Values('regularization', ['dropout', 'l1', 'l2']),
                b=ValueRange('regularization_coefficient', [0, 1])),
            Req(a=Values('regularization', ['dropout']),
                b=Values('regularization_application', ['recurrent', None])),
            Req(a=Values('regularization', ['l1', 'l2']),
                b=Values('regularization_application', ['recurrent', 'bias', 'activity']))
        ]

        for req in requirements:
            req.validate(self.args)
