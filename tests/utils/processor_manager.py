import json
import os
import signal
import atexit
from importlib.util import find_spec


class NoProcessorAvailableError(Exception):
    """
    All processors on the host are in use
    """
    pass

class ProcessorManager():
    """
    Class used for running multiple machine learning models at once, using separate processors.

    The manager is a class that should be run with each model being fit. It keeps a list of available processors on
    disk, and checks one out for a model (again, writing to disk). When the model is done running, the processor is
    checked back in.

    Treats each GPU as a unique processor, but all CPUs as a single processor
    """
    def __init__(self, save_dir='/home/isaac/data/', debug=False):
        self.processor_list_fp = os.path.join(save_dir, '.processor_list.json')
        self.my_processor = None
        self.debug = debug

        self.version = 'torch' if find_spec('torch') is not None else 'tensorflow'

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def log_sigint(self, a=None, b=None):
        """
        Close the manager if the model is terminated early by signal interuption

        Gets fed two inputs when called, neither of which are needed

        :param a: Not used
        :param b: Not used
        :return: None
        """
        self.close()

    def _add_exit_handling(self):
        """
        Add the log_sigint function to the exit handling process

        :return: None
        """
        signal.signal(signal.SIGINT, self.log_sigint)
        atexit.register(self.log_sigint)

    def _load(self):
        """
        Load the list of processors from disk

        :return: None
        """
        if os.path.exists(self.processor_list_fp):
            with open(self.processor_list_fp, "r") as outfile:
                self.data = json.load(outfile)
        else:
            self._create_processor_list()

    def _remove_most_recent(self):
        """
        Check the most recently checked-out processor back in

        Useful when the most recent run was terminated without a proper exit handle

        :return: None
        """
        self._load()
        if len(self['in_use']) > 0:
            self['in_use'] = self['in_use'][:-1]
        self._save()

    def _remove_all(self):
        """
        Check back in all processors that have been checked out

        Useful if a bunch of processors were terminated without a proper exit handle (e.g. if the host lost power
        suddenly)

        :return: None
        """
        self._create_processor_list(overwrite=True)
        self._save()

    def _save(self):
        """
        Save the current list of processors (including those that are checked out and in) to disk

        :return: None
        """
        with open(self.processor_list_fp, "w") as outfile:
            json.dump(self.data, outfile)

    def _create_processor_list(self, overwrite=False):
        """
        Create list of processors that can be used

        If tensorflow is available, uses tensorflow's list of devices to create list. Otherwise, uses pytorch. Treats
        each GPU as it's own processor and all CPUs as a single processor.
        """
        if not overwrite:
            if os.path.exists(self.processor_list_fp):
                raise FileExistsError('Processor list already exists. It should not be recreated as it may contain'
                                      'information on which processors are currently in use')
        if self.version == 'tensorflow':
            import tensorflow as tf
            self.data = {}
            processors = [dv.name for dv in tf.config.list_logical_devices()]
            processors = [p for p in processors if 'XLA' not in p]
        elif self.version == 'torch':
            from torch.cuda import device_count
            num_gpus = torch.cuda.device_count()
            processors = [f'/device:GPU:{i}' for i in range(num_gpus)]
            processors.append(f'/device:CPU:0')

        self.data['processors'] = processors
        self.data['in_use'] = []

    def _choose_device(self):
        """
        Select the device that should be used by the model

        :return: Processor to use
        :rtype: str
        """
        # If we're debugging, use CPU, since even if CPU is already in use, room can still be made available
        if self.debug:
            all_cpus = [p for p in self.data['processors'] if 'CPU' in p]
            return all_cpus[0]
        else:
            available_gpus = [p for p in self.data['processors'] if 'GPU' in p and p not in self.data['in_use']]
            available_cpus = [p for p in self.data['processors'] if 'CPU' in p and p not in self.data['in_use']]
            if len(available_gpus) > 0:
                return available_gpus[0]
            elif len(available_cpus) > 0:
                return available_cpus[0]
            else:
                raise NoProcessorAvailableError(f'All processors currently in use. These are {self.data["processors"]}')

    def open(self):
        """
        Check out a processor

        :return: None
        """
        if self.my_processor is not None:
            print(f'Processor already open. The processor being used is {self.my_processor}')
        else:
            self._load()
            self.my_processor = self._choose_device()
            if not self.debug:
                self.data['in_use'].append(self.my_processor)
                self._save()
                self._add_exit_handling()
            if 'cpu' in self.my_processor.lower() and self.version == 'tensorflow':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            elif self.version == 'tensorflow':
                os.environ['CUDA_VISIBLE_DEVICES'] = self.device()[-1]

    def close(self):
        """
        Check the processor back in

        :return: None
        """
        if self.my_processor is None:
             print(f'Processor is not open.')
        else:
            self._load()
            if not self.debug:
                self.data['in_use'] = [p for p in self.data['in_use'] if p != self.my_processor]
                self._save()
                atexit.unregister(self.log_sigint)
            self.my_processor = None

    def device(self):
        """
        Return the processor that this manager has checked out

        :return: The processor that this manager has checked out
        :rtype: str
        """
        return self.my_processor


if __name__ == '__main__':
    mgr = ProcessorManager()
    mgr._remove_all()
