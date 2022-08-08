# This class has been deprecated, and no longer works with all configurations
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
import os
from config import config
import urllib
import keras
from mlflow import log_metric

class MedianStopper(keras.callbacks.Callback):
    """
    Based on description in section 3.2.2 of the below paper:
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf

    Calculates, for each model that has been fit, at each step, what its average loss was.
    If for a given step, the current model's best loss is worse than the median average loss, the model
    is ended early.
    """
    def __init__(self, run_id, experiment_id, iterations = 30):
        self.cutoffs_path = os.path.join(config.box_and_year_dir, '.median_losses.csv')
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.early_stopping_occured = False
        self.stopped_epoch = -1
        self.best_loss = np.Inf
        self.iterations = iterations

    def on_train_begin(self, logs=None):
        """
        Load the median average loss from previous training runs when training begins

        These will be used as the cutoffs for early stopping

        :param logs: Information on current run
        :return: None
        """
        self._load_cutoffs()

    def on_train_end(self, logs=None):
        """
        Record whether early stopping occurred or not in the console and logs

        :param logs: Information on current run
        :return: None
        """
        if self.early_stopping_occured:
            print(f'Epoch {self.stopped_epoch + 1}: early stopping occurred based on median loss')
            log_metric('early_median_stopping_occured', 1.)
        else:
            print(f'Early stopping did not occur based on median loss')
            log_metric('early_median_stopping_occured', 0.)

    def on_epoch_end(self, epoch, logs=None):
        """
        Log the best loss after each epoch, and if this is an iteration to check for early stopping, do so

        :param epoch: The epoch number
        :param logs: Information on current run
        :return: None
        """
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        if ((epoch + 1) / self.iterations) % 1 == 0:
            if epoch + 1 in self.cutoffs['steps'].values:
                cutoff = self.cutoffs[self.cutoffs['steps'] == epoch + 1]['average_loss'].iloc[0]
            else:
                cutoff = np.Inf
            if np.greater(self.best_loss, cutoff):
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.early_stopping_occured = True

    def _load_cutoffs(self):
        """
        Load the median average loss from previous training runs


        :return: None
        """
        self.cutoffs = pd.read_csv(self.cutoffs_path)
        self.cutoffs = self.cutoffs[self.cutoffs['experiment_id'] == int(self.experiment_id)]

    def recalculate_cutoffs(self):
        """
        After training ends, recalculate the median mean losses across the models

        Writes the medians so they can be easily loaded by the next run

        :return: None
        """
        client = MlflowClient()
        experiments = [exp.experiment_id for exp in client.list_experiments()]
        runs = client.search_runs(
            experiment_ids=experiments,
            run_view_type=ViewType.ACTIVE_ONLY
        )

        run_info = pd.DataFrame(columns=['steps','average_loss','experiment_id'])
        for run in runs:
            if run.info.status == 'FINISHED' or run.info.run_id == self.run_id:
                artifact_dir = run.info._artifact_uri
                overall_dir = os.path.dirname(artifact_dir)
                try:
                    loss_path = os.path.join(overall_dir, 'metrics', 'epoch_val_loss')
                    loss_history = pd.read_csv(loss_path,names=['time','loss','step'], sep='\s+')
                except urllib.error.URLError:
                    loss_path = os.path.join(overall_dir, 'metrics', 'val_loss')
                    loss_history = pd.read_csv(loss_path,names=['time','loss','step'], sep='\s+')
                loss_history['steps'] = 1
                loss_history = loss_history[['loss','steps']].cumsum()
                loss_history['average_loss'] = loss_history['loss'] / loss_history['steps']

                loss_history = loss_history.drop(['loss'],axis=1)
                loss_history['experiment_id'] = run.info.experiment_id

                run_info = pd.concat([run_info, loss_history])
        self.cutoffs = run_info.groupby(['experiment_id','steps']).median()

        self.cutoffs.to_csv(self.cutoffs_path)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

if __name__ == '__main__':
    stopper = MedianStopper(run_id=None, experiment_id='6')
    stopper.recalculate_cutoffs()

