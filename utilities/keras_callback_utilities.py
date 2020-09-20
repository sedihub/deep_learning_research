"""A collection of callbacks
"""

import tensorflow as tf

class CustomHistory(tf.keras.callbacks.Callback):
    """A custom callback to monitor model training.
    """
    def __init__(self, *args, **kwargs):
        super(CustomHistory, self).__init__(*args, **kwargs)
        self._epochs = None
        self._current_epoch = None
        self._loss_key = "loss"
        
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        if(self._loss_key not in keys):  # If `loss` is not there, find the most likely candidate
            for key in keys:
                if(key.find("loss") != -1):
                    self._loss_key = key
                    break
        self._epochs_losses = {}
    
    def _raise_error_on_none_logs(self):
        raise ValueError("\"logs\" is None!")

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        if(logs is None):
            _raise_error_on_none_logs
        self._current_epoch = epoch
        self._epochs_losses[epoch] = []

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self._epochs_losses[self._current_epoch].append(
            logs[self._loss_key])
    
    def get_loss_history(self,):
        return self._epochs_losses