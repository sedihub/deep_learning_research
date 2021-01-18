""" A set of utility function for the notebook.
"""

import tensorflow as tf
import numpy as np
from scipy.special import softmax


class Scrambler():
    def __init__(self, target_shape):
        self._target_shape = target_shape
        self._initialize_scramble()
        
    def _initialize_scramble(self):
        self._scrambling_array = np.arange(np.prod(self._target_shape))
        np.random.shuffle(self._scrambling_array)
        self._scrambling_array_inverse = np.argsort(self._scrambling_array)
        
    def scramble(self, x):
        x_shape = x.shape
        if np.prod(x[0].shape) == np.prod(self._target_shape):
            return x.reshape(
                (x_shape[0], self._scrambling_array.shape[0])
            )[:, self._scrambling_array].reshape(x_shape)
        elif np.prod(x.shape) == np.prod(self._target_shape):
            return x.flatten()[self._scrambling_array].reshape(x_shape)
        else:
            raise ValueError(f"The input shape, {x.shape}, is not valid!")
    
    def unscramble(self, x):
        x_shape = x.shape
        if np.prod(x[0].shape) == np.prod(self._target_shape):
            return x.reshape(
                (x_shape[0], self._scrambling_array.shape[0])
            )[:, self._scrambling_array_inverse].reshape(x_shape)
        elif np.prod(x.shape) == np.prod(self._target_shape):
            return x.flatten()[self._scrambling_array_inverse].reshape(x_shape)
        else:
            raise ValueError(f"The input shape, {x.shape}, is not valid!")
        
    def set_scrambling_array(self, scrambing_array, copy=True):
        if scrambing_array.shape != self._scrambling_array.shape:
            raise ValueError(f"The array does not have the right shape:\n\
            {scrambing_array.shape} vs. {self._scrambling_array.shape}!")
        if copy:
            self._scrambling_array = np.copy(scrambing_array)
            self._scrambling_array_inverse = np.argsort(self._scrambling_array)
        else:
            self._scrambling_array = scrambing_array
            self._scrambling_array_inverse = np.argsort(self._scrambling_array)
        
    def get_scrambling_array(self, copy=True):
        if copy:
            return np.copy(self._scrambling_array)
        else:
            return self._scrambling_array


class MNIST_Sequence_Generator(tf.keras.utils.Sequence):
    """A sequence generator for the MNIST task.
    """
    def __init__(self, x_array, y_array, batch_size):
        self.x, self.y = x_array, y_array
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return (batch_x, batch_y)
    
    
class MNIST_or_Noise_Sequence_Generator(tf.keras.utils.Sequence):
    """A sequence generator fake or real MNIST image.
    """
    def __init__(self, x_array, fake_x_array, batch_size):
        self.x = x_array
        self.fake_x = fake_x_array
        self.batch_size = batch_size
        #
        self._half_length = len(self.x) // self.batch_size
        self._length = 2 * self._half_length
        #
        self._choose = np.random.choice(
            a=[True, False], size=(self._length,), p=[0.5, 0.5])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._choose[idx]:
            idx = idx % self._half_length
            batch_x = self.x[idx * self.batch_size:(idx  + 1) * self.batch_size]
            batch_y = np.ones(shape=(self.batch_size,), dtype=self.x.dtype)
        else:
            idx = idx % self._half_length
            batch_x = self.fake_x[idx * self.batch_size:(idx  + 1) * self.batch_size]
            batch_y = np.zeros(shape=(self.batch_size,), dtype=self.x.dtype)
        return (batch_x, batch_y)
    

class MNIST_GAN_Sequence_Generator(tf.keras.utils.Sequence):
    """A sequence generator fake or real MNIST image.
    """
    def __init__(self, x_array, batch_size, hidden_rep_size=10, temperature=None):
        self.x = x_array
        self.batch_size = batch_size
        self.temperature = temperature
        self.hidden_rep_size = hidden_rep_size
        #
        self._length = 2 * (len(self.x) // self.batch_size)
        #
        self._choose = np.random.choice(
            a=[True, False], size=(self._length,), p=[0.5, 0.5])
        
        self._true_array = np.array([True]*self.batch_size, dtype=bool)
        self._false_array = np.array([False]*self.batch_size, dtype=bool)
        
        image_shape = (batch_size, *self.x.shape[1:]) 
        self._zero_image = np.zeros(image_shape, dtype=self.x.dtype)
        self._zero_prob = np.zeros((batch_size, 10), dtype=np.float32)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._choose[idx]:
            idx = idx // 2
            x_1 = self._true_array
            x_2 = self.x[idx * self.batch_size:(idx  + 1) * self.batch_size]
            x_3 = self._zero_prob
            y = np.ones(shape=(self.batch_size,), dtype=np.int32)
        else:
            x_1 = self._false_array
            x_2 = self._zero_image
            if self.temperature is None:
                x_3 = np.random.randn(self.batch_size, self.hidden_rep_size)
            else:
                x_3 = softmax(np.random.randn(self.batch_size, self.hidden_rep_size) * self.temperature, axis=1)
            y = np.zeros(shape=(self.batch_size,), dtype=np.int32)
        x = (x_1, x_2, x_3)
        return (x, y)
    