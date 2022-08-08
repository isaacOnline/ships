import argparse
import logging
import os
import dask

import dask.dataframe as dd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from haversine import haversine_vector
from config import config
from processing_step import ProcessingStep
from utils import clear_path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from config.dataset_config import datasets

class DestinationAppender(ProcessingStep):
    """
    Class for appending a destination cluster to each trajectory
    """
    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name='interpolated' + ('_debug' if args.debug else ''),
            to_name='interpolated_with_destination' + ('_debug' if args.debug else '')
        )
        self.EARTH_RADIUS = 6371 # in km

        self._initialize_logging(args.save_log, 'add_destination')

        self.artifact_directory = os.path.join(self.artifact_directory, 'destination_appending' + ('_debug' if args.debug else ''))
        if not os.path.exists(self.artifact_directory):
            os.mkdir(self.artifact_directory)


    def load(self):
        """
        Load the test, train, and validation sets

        This function just specifies the paths for dask. (Dask uses lazy evaluation so the full sets aren't read in
        here.)

        :return:
        """
        for dataset_name in ['test', 'valid', 'train']:
            dataset_path = os.path.join(self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)

        logging.info('File paths have been specified for dask')


    def save(self):
        """
        Save the processed datasets to dask.

        Because Dask uses lazy evaluation, the processing will actually happen only when this method is called.

        :return:
        """

        for dataset_name in ['test','valid','train']:
            out_path = os.path.join(self.to_dir, f'{dataset_name}.parquet')
            clear_path(out_path)
            dd.to_parquet(self.datasets[dataset_name], out_path, schema='infer')
            logging.info(f'{dataset_name} set saved')


    def plot_nn_elbows(self, final_messages):
        """
        Plot a sorted "bar" chart of nearest neighbors

        This helps the user decide on a range of epsilon values to try. The range should be chosen to cover the
        "elbow" in the plot. See https://dl.acm.org/doi/pdf/10.1145/3068335 for more info.

        """
        final_messages = final_messages[['lat_rad','lon_rad']]
        n_neighbors_to_try = np.array(config.dataset_config.min_pts_to_try) - 1

        n_neighbors = np.max(n_neighbors_to_try)

        neighbor_calculator = NearestNeighbors(n_neighbors=n_neighbors, metric='haversine')
        neighbor_calculator.fit(final_messages)
        dist, _ = neighbor_calculator.kneighbors(final_messages)

        dist *= self.EARTH_RADIUS

        plt.clf()
        for n_neighbors in n_neighbors_to_try:
            dist_to_last_neighbor = dist[:, n_neighbors - 1]
            dist_to_last_neighbor.sort()
            dist_to_last_neighbor = np.flip(dist_to_last_neighbor)

            plt.plot(dist_to_last_neighbor, label = f'Distance to {n_neighbors} NN (minPts = {n_neighbors+1})')

        plt.legend()
        plt.ylabel('Distance (KM)')
        plt.xlabel('Sorted Point Index')
        plt.minorticks_on()
        plt.grid(visible=True, which='both', axis='y')
        save_path = os.path.join(self.artifact_directory, f'nn_distance.pdf')
        plt.savefig(save_path)
        plt.clf()

    def _plot_n_clusters(self, min_pts, n_clusters):
        """
        Plot a line plot of the number of clusters for different epsilon values, for a set value of min_pts

        :param min_pts: The specified min_pts value
        :param n_clusters: The calculated number of clusters for each eps
        :return:
        """
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(config.dataset_config.eps_to_try, n_clusters,
                 linestyle='-', marker='o')

        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Number of Unique Clusters')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        save_dir = os.path.join(self.artifact_directory, str(min_pts))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fig.savefig(os.path.join(save_dir,'n_clusters.pdf'))

    def _plot_cluster_size(self, min_pts, avg_cluster_sizes, avg_unique_mmsis_per_cluster):
        """
        Plot the average cluster sizes for a set value of min_pts

        :param min_pts: Set value of min_pts
        :param avg_cluster_sizes: Pre-calculated values
        :param avg_unique_mmsis_per_cluster: Pre-calculated values
        :return:
        """
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(config.dataset_config.eps_to_try, avg_cluster_sizes,
                 linestyle='-', marker='o', label = 'Avg Number of Tracks per Cluster')
        ax.plot(config.dataset_config.eps_to_try, avg_unique_mmsis_per_cluster,
                 linestyle='-', marker='o',label = 'Avg Number of MMSIs per Cluster')

        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Avg per Cluster')
        ax.legend()
        ax.set_title('Cluster Sizes')
        save_dir = os.path.join(self.artifact_directory, str(min_pts))
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir,'cluster_sizes.pdf'))

    def _plot_cluster_makeup(self, min_pts, pct_in_largest_cluster, pct_of_tracks_in_clusters):
        """
        Plot the percent of tracks in the largest cluster and the percent of tracks in any cluster for a set value of
        min_pts


        :param min_pts: Set value of min_pts
        :param pct_in_largest_cluster: Pre-calculated values
        :param pct_of_tracks_in_clusters: Pre-calculated values
        :return:
        """
        plt.clf()
        plt.plot(config.dataset_config.eps_to_try, pct_of_tracks_in_clusters,linestyle='-', marker='o', label='Pct of Tracks in any Cluster')
        plt.plot(config.dataset_config.eps_to_try, pct_in_largest_cluster,linestyle='-', marker='o', label='Pct of Tracks in Largest Cluster')

        save_dir = os.path.join(self.artifact_directory, str(min_pts))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        plt.ylabel('Share of dataset')
        plt.xlabel('Epsilon')
        plt.legend()
        plt.title('Cluster Makeup')
        plt.savefig(os.path.join(save_dir,'cluster_makeup.pdf'))

    def _plot_cluster_geographic_sizes(self, min_pts, avg_sq_kms):
        """
        Plot the average size of the bounding box defining each cluster for a set value of min_pts

        :param min_pts: Set value of min pts
        :param avg_sq_kms: Pre-calculated values
        :return:
        """
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(config.dataset_config.eps_to_try, avg_sq_kms, linestyle='-', marker='o')

        ax.set_title("Average Size of Clusters' Bounding Boxes")
        ax.set_ylabel('Square KM')
        ax.set_xlabel('Epsilon')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())

        save_dir = os.path.join(self.artifact_directory, str(min_pts))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fig.savefig(os.path.join(save_dir,'geographic_sizes.pdf'))

    def plot_cluster_info(self, final_messages, nn_distances):
        """
        Plot a number of relevant cluster quality metrics for the different specified values of eps

        Plots the percent of tracks that are in the noise cluster, the percent of tracks that are in the largest cluster,
        the average number of tracks per cluster, the average number of unique MMSIs per cluster, the average geographic
        sizes of each cluster, and the number of unique clusters. These should help the user select a desired
        min_pts to use.

        :param final_messages:
        :param nn_distances:
        :return:
        """
        for min_pts in config.dataset_config.min_pts_to_try:
            n_clusters = []
            pct_of_tracks_in_clusters = []
            pct_in_largest_cluster = []
            avg_cluster_pcts = []
            avg_cluster_sizes = []
            avg_unique_mmsis_per_cluster = []
            avg_sq_kms = []
            for km_eps in config.dataset_config.eps_to_try:
                eps = km_eps / self.EARTH_RADIUS
                db = DBSCAN(eps=eps, min_samples=min_pts, metric='precomputed').fit(nn_distances)
                labels = pd.Series(db.labels_)

                # Number of clusters
                in_clusters = labels[labels != -1]
                n_clusters += [in_clusters.nunique()]

                # Share of points that are in a cluster
                pct_of_tracks_in_clusters += [len(in_clusters)/len(labels)]

                # Share of points that are in the largest cluster
                cluster_counts = pd.Series(in_clusters).value_counts()
                if len(cluster_counts) > 0:
                    n_in_largest = cluster_counts.iloc[0]
                else:
                    n_in_largest = 0
                pct_in_largest_cluster += [n_in_largest / len(labels)]

                # Avg cluster size
                avg_cluster_sizes += [cluster_counts.mean()]
                pct_in_cluster = cluster_counts / len(labels)
                avg_cluster_pcts += [pct_in_cluster.mean()]

                # Average number of MMSIs per cluster
                labels_and_mmsis = pd.DataFrame({'mmsi': final_messages['mmsi'], 'label': labels})
                labels_and_mmsis = labels_and_mmsis[labels_and_mmsis['label'] != -1]
                labels_and_mmsis = labels_and_mmsis.groupby('label').agg(['count', 'nunique'])
                avg_unique_mmsis_per_cluster += [labels_and_mmsis[('mmsi','nunique')].mean()]

                # Average cluster width/ length
                labels_and_coords = pd.DataFrame({'lat': final_messages['lat'],
                                                  'lon': final_messages['lon'],
                                                  'label': labels})
                labels_and_coords = labels_and_coords[labels_and_coords['label'] != -1]
                labels_and_coords = labels_and_coords.groupby('label').agg(['min','max'])
                labels_and_coords['x'] = haversine_vector(labels_and_coords[[('lat', 'min'), ('lon', 'min')]], labels_and_coords[[('lat', 'min'), ('lon', 'max')]])
                labels_and_coords['y'] = haversine_vector(labels_and_coords[[('lat', 'min'), ('lon', 'min')]], labels_and_coords[[('lat', 'max'), ('lon', 'min')]])
                labels_and_coords['sq_km'] = labels_and_coords['x'] * labels_and_coords['y']
                avg_sq_kms += [labels_and_coords['sq_km'].mean()]

            self._plot_n_clusters(min_pts, n_clusters)
            self._plot_cluster_makeup(min_pts, pct_in_largest_cluster, pct_of_tracks_in_clusters)
            self._plot_cluster_size(min_pts, avg_cluster_sizes, avg_unique_mmsis_per_cluster)
            self._plot_cluster_geographic_sizes(min_pts, avg_sq_kms)

    def _sort_graph(self, graph):
        """
        Sort a nearest neighbor graph, so the sorting doesn't need to be done by sklearn repeatedly

        Taken from the sklearn dbscan fn, which does this sorting itself. I do it here so I only have to do it once

        :param graph: nearest neighbors graph
        :return:
        """
        row_nnz = np.diff(graph.indptr)
        if row_nnz.max() == row_nnz.min():
            n_samples = graph.shape[0]
            distances = graph.data.reshape(n_samples, -1)

            order = np.argsort(distances, kind="mergesort")
            order += np.arange(n_samples)[:, None] * row_nnz[0]
            order = order.ravel()
            graph.data = graph.data[order]
            graph.indices = graph.indices[order]

        else:
            for start, stop in zip(graph.indptr, graph.indptr[1:]):
                order = np.argsort(graph.data[start:stop], kind="mergesort")
                graph.data[start:stop] = graph.data[start:stop][order]
                graph.indices[start:stop] = graph.indices[start:stop][order]

    def _calculate_nn_distances(self, messages):
        """
        Find the nearest neighbors for AIS messages based on haversine distance

        :param messages: AIS messages, containing latitude/longitude in radians
        :return:
        """
        distances = radius_neighbors_graph(messages[['lat_rad', 'lon_rad']],
                                           radius=np.max(config.dataset_config.eps_to_try) / self.EARTH_RADIUS,
                                           metric='haversine', mode='distance')
        self._sort_graph(distances)
        return distances

    def create_parameter_plots(self):
        """
        Create plots to help with deciding DBSCAN parameters

        If the user has specified min_pts_to_try only, this will create elbow plots, similar to those on page
        19:11 of https://dl.acm.org/doi/pdf/10.1145/3068335. This will help the user identify a range of epsilon values
        to try.

        Once the user has specified eps_to_try, this will go through, and for each min_pts value specified in
        min_pts_to_try, plot a number of relevant cluster quality metrics for all of the different specified values of
        eps: The percent of tracks that are in the noise cluster, the percent of tracks that are in the largest cluster,
        the average number of tracks per cluster, the average number of unique MMSIs per cluster, the average geographic
        sizes of each cluster, and the number of unique clusters. These should help the user select a desired
        min_pts to use.

        :return:
        """
        final_messages = self.load_final_messages('train', max_samples=75000)

        if config.dataset_config.eps_to_try is not None:
            nn_distances = self._calculate_nn_distances(final_messages)
            self.plot_cluster_info(final_messages, nn_distances)
        elif config.dataset_config.min_pts_to_try is not None:
            self.plot_nn_elbows(final_messages)
        else:
            raise ValueError('Must at least specify min_pts_to_try in dataset config in order to run '
                             'destination_appender')

    def create_point_plot(self, final_messages, eps):
        """
        Plot the destination clusteers that were found

        :param final_messages:
        :param eps:
        :return:
        """
        groups = final_messages.groupby('label')
        plt.clf()

        colors = [c for c in mcolors.CSS4_COLORS.keys() if 'grey' not in c and 'gray' not in c and 'white' not in c]
        i = 0
        for name, group in groups:
            if name == '-1':
                plt.scatter(group['lon'], group['lat'], marker='o', s=0.1,
                         label='Not Clustered', c='gray')
            else:
                plt.scatter(group['lon'], group['lat'], marker='o', s=0.1,
                            c = colors[i])
                i += 1
                i %= len(colors)
        plt.legend()
        save_path = os.path.join(self.artifact_directory, f'{config.dataset_config.min_pts_to_use}_'
                                                          f'{eps}_'
                                                          f'{config.dataset_config.dataset_name}.pdf')
        plt.savefig(save_path)
        pct_in_largest_cluster = final_messages['label'][final_messages['label'] != '-1'].value_counts().iloc[0] / len(final_messages)
        logging.info(f'For minPts={config.dataset_config.min_pts_to_use} and epsilon={eps}, '
                     f'{pct_in_largest_cluster*100:0.4}% of trajectories are '
                     f'in the largest cluster')

        pct_noise = (final_messages['label'] == '-1').mean()
        logging.info(f'{pct_noise*100:0.4}% of trajectories were unable to be clustered and are considered noise')

        num_clusters = final_messages['label'][final_messages['label']!= '-1'].nunique()
        logging.info(f'There were {num_clusters:,} clusters found')

        avg_cluster_size = final_messages['label'][final_messages['label']!= '-1'].value_counts().mean()
        logging.info(f'On average, there were {avg_cluster_size:0.5} trajectories in each cluster')

        labels_and_coords = final_messages[final_messages['label'] != '-1']
        labels_and_coords = labels_and_coords.groupby('label').agg(['min', 'max'])
        labels_and_coords['x'] = haversine_vector(labels_and_coords[[('lat', 'min'), ('lon', 'min')]],
                                                  labels_and_coords[[('lat', 'min'), ('lon', 'max')]])
        labels_and_coords['y'] = haversine_vector(labels_and_coords[[('lat', 'min'), ('lon', 'min')]],
                                                  labels_and_coords[[('lat', 'max'), ('lon', 'min')]])
        labels_and_coords['sq_km'] = labels_and_coords['x'] * labels_and_coords['y']
        avg_cluster_geographic_size = labels_and_coords['sq_km'].mean()

        logging.info(f'On average, the bounding box for each cluster was {avg_cluster_geographic_size:0.6} '
                     f'square kilometers')

    def fit_dbscan(self, final_messages, eps_to_use):
        """
        Perform DBSCAN clustering for AIS messages

        :param final_messages: Dataset containing lat/lon coordinates
        :param eps_to_use: Neighborhood radius parameter for dbscan
        :return:
        """
        nn_distances = self._calculate_nn_distances(final_messages)

        eps = eps_to_use / self.EARTH_RADIUS
        db = DBSCAN(eps=eps, min_samples=config.dataset_config.min_pts_to_use, metric='precomputed').fit(nn_distances)
        final_messages['label'] = pd.Series(db.labels_).astype(str)
        return final_messages

    def load_final_messages(self, dataset, max_samples=None):
        end_times = self.datasets[dataset].groupby('track')[['base_datetime']].max().reset_index()
        final_messages = self.datasets[dataset].merge(end_times, on=['track','base_datetime'])[['lat','lon','mmsi']].compute().reset_index(drop=True)
        final_messages['lat_rad'] = np.radians(final_messages['lat'])
        final_messages['lon_rad'] = np.radians(final_messages['lon'])
        if max_samples is not None:
            if len(final_messages) > max_samples:
                np.random.seed(4148846)
                final_messages = final_messages.sample(max_samples).reset_index(drop=True)
        return final_messages

    def fit_classifier_model(self, final_messages):
        """
        Fit a classifier in order to predict the clusters that have been found using DBSCAN

        :param final_messages: Labeled final AIS messages from trajectories
        :return:
        """
        TEST_SIZE = 1000
        test_idxs = final_messages.sample(TEST_SIZE).index
        test = final_messages.loc[test_idxs]
        train = final_messages[~final_messages.index.isin(test_idxs)]
        self.model = RandomForestClassifier(n_estimators=100,
                                            criterion = 'gini',
                                            min_samples_split=2, min_samples_leaf=1, max_depth = 25,
                                            max_features = 2,
                                            bootstrap=True, max_samples = 0.8,
                                            random_state=83845640,
                                            oob_score=True)
        self.model = self.model.fit(train[['lat', 'lon']], train['label'])
        test_preds = self.model.predict(test[['lat','lon']])
        num_correct = (test_preds == test['label']).sum()
        incorrect_labels = test['label'][test_preds != test['label']]
        true_vals_for_incorrect_labels = test_preds[test_preds != test['label']]

        num_on_border = ((incorrect_labels == '-1') |  (true_vals_for_incorrect_labels == '-1')).sum()

        logging.info(f'For the destination classification model, {TEST_SIZE:,} rows were used for the test set and '
                     f'{len(train):,} rows were used for the training set. The OOB score from the model was '
                     f'{self.model.oob_score_}. Of the {TEST_SIZE:,} messages in the test set, {num_correct} were '
                     f'labeled correctly. Of the messages that were incorrectly labeled, {num_on_border} were either '
                     f'incorrectly labeled as unclustered, or were unclustered in reality but were incorrectly labeled '
                     f'as part of a cluster.')

    def predict_weather_partition(self, partition, model, cluster_centers):
        """
        Use the trained destination classifier to label AIS messages

        :param partition: Dataset to label
        :param model: Classifier object
        :param cluster_centers: The mean latitude/longitude for each cluster
        :return:
        """
        # Get the final messages for the partition
        end_times = partition.groupby('track')[['base_datetime']].max().reset_index()
        final_messages = partition.merge(end_times, on=['track', 'base_datetime'])[
            ['lat', 'lon','track']].reset_index(drop=True)

        # Use the classifier to label
        final_messages['destination_cluster'] = model.predict(final_messages[['lat','lon']]).astype(int)

        # Merge the labels for the final messages into *all* messages from the trajectory
        final_messages = final_messages[['track','destination_cluster']]
        partition = partition.merge(final_messages, left_index=True, right_on='track')

        # Add information about where the destination clusters are centered
        cluster_centers.index = cluster_centers.index.astype(int)
        partition = partition.merge(cluster_centers, left_on = 'destination_cluster', right_index=True)

        # Return original index/sorting
        partition = partition.set_index('track')
        partition = partition.sort_values(['track','base_datetime'])
        partition['destination_cluster'] = partition['destination_cluster'].astype(str)
        return partition

    def plot_clusters(self):
        """
        Plots the clusters found for each of the eps_to_try values chosen

        Once the min_pts_to_use value has been specified, this can help the user visualize the clusters that occur for
        different values of eps.

        :return:
        """
        final_messages = self.load_final_messages('train', max_samples = 75000)

        for eps in config.dataset_config.eps_to_try:
            labeled_final_messages = self.fit_dbscan(final_messages, eps)
            self.create_point_plot(labeled_final_messages, eps)


    def append(self):
        """
        Append the destination clusters to the AIS messages

        Once the eps and min_pts values for a dataset have been selected, this goes through and finds the clusters using
        the specified params. Because everything up until now has been done on the training set, and DBSCAN is a
        clustering algorithm (not a classification algorithm), a classifier is then fit in order to label the final
        messages.

        :return:
        """
        final_messages = self.load_final_messages('train', max_samples = 75000)
        final_messages = self.fit_dbscan(final_messages, config.dataset_config.eps_to_use)
        cluster_centers = final_messages.groupby('label')[['lat', 'lon']].mean()
        cluster_centers.columns = ['destination_cluster_lat_center','destination_cluster_lon_center']
        self.create_point_plot(final_messages, config.dataset_config.eps_to_use)
        self.fit_classifier_model(final_messages)


        partition = self.datasets['test'].partitions[0].compute()
        output_meta = self.predict_weather_partition(partition, self.model, cluster_centers)
        output_meta = output_meta.iloc[0:0]

        # Go through and label each of the datasets
        for dataset_name in ['train','test','valid']:
            self.datasets[dataset_name] = self.datasets[dataset_name].map_partitions(self.predict_weather_partition,
                                                                                     meta=output_meta,
                                                                                     model=self.model,
                                                                                     cluster_centers=cluster_centers
                                                                                     )





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())

    # Logging and debugging
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    config.set_log_level(args.log_level)
    config.dataset_config = datasets[args.dataset_name]

    if args.debug:
        dask.config.set(scheduler='single-threaded')
    else:
        dask.config.set(scheduler='single-threaded')

    appender = DestinationAppender()
    appender.load()
    if config.dataset_config.min_pts_to_use is not None and config.dataset_config.eps_to_use is not None:
        appender.append()
        appender.save()
    elif config.dataset_config.min_pts_to_use is not None:
        appender.plot_clusters()
    else:
        appender.create_parameter_plots()
