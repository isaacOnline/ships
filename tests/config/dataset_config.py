class DatasetConfig():
    def __init__(self, dataset_name,
                 lat_1, lat_2, lon_1, lon_2,
                 sliding_window_movement,
                 depth_1=0, depth_2=0,
                 min_pts_to_try = None, eps_to_try = None,
                 min_pts_to_use = None, eps_to_use = None):
        self.dataset_name = dataset_name
        self.lat_1 = min(lat_1, lat_2)
        self.lat_2 = max(lat_1, lat_2)
        self.lon_1 = min(lon_1, lon_2)
        self.lon_2 = max(lon_1, lon_2)
        self.corner_1 = (lat_1, lon_1)
        self.corner_2 = (lat_2, lon_2)
        # When expanding data using a sliding window, the sliding_window_movement is the length of time between windows.
        # E.g. if sliding_window_length is 10 * 60, tracks are supposed to be made up of three timestamps, and the
        # interpolated trajectory has timestamps at [0, 5, 10, 15, 20], then the trajectories output
        # will be [0, 5, 10], and [10, 15, 20]
        self.sliding_window_movement = sliding_window_movement
        self.min_pts_to_try = min_pts_to_try
        self.eps_to_try = eps_to_try
        self.min_pts_to_use = min_pts_to_use
        self.eps_to_use = eps_to_use
        self.depth_1 = depth_1
        self.depth_2 = depth_2



datasets = {
    'florida_gulf':
        DatasetConfig(
            dataset_name='florida_gulf',
            lat_1=26.00, lon_1=-85.50, lat_2=29.00, lon_2=-81.50,
            min_pts_to_try=[4, 10, 20, 50, 100, 250, 500],
            eps_to_try=[0.0001, 0.00025, 0.0005, 0.00075,
                        0.001, 0.0025, 0.005, 0.0075,
                        0.01, 0.025, 0.05, 0.075,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                        1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                        6, 7, 8, 9, 10],
            min_pts_to_use=50,
            eps_to_use=5,
            sliding_window_movement=15 * 60
        ),
    'california_coast':
        DatasetConfig(
            dataset_name='california_coast',
            lat_1=33.40, lon_1=-122.00, lat_2=36.40, lon_2=-118.50,
            sliding_window_movement=15 * 60,
            min_pts_to_try = [4, 10, 20, 50, 100, 250, 500],
            eps_to_try=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
            min_pts_to_use = 50,
            eps_to_use = 3
        ),
    'new_york':
        DatasetConfig(
            dataset_name='new_york',
            lat_1=39.50, lon_1=-74.50, lat_2=41.50, lon_2=-71.50,
            sliding_window_movement=60 * 60,
            min_pts_to_try=[4, 10, 20, 50, 100, 250, 500],
            eps_to_try=[0.0001, 0.00025, 0.0005, 0.00075,
                        0.001, 0.0025, 0.005, 0.0075,
                        0.01, 0.025, 0.05, 0.075,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                        1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                        6, 7, 8, 9, 10],
            min_pts_to_use=50, eps_to_use=3
        )
}
