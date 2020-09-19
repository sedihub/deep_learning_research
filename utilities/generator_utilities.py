"""A Set of data generators.
"""

import tensorflow as tf
import numpy as np


class ScrambledImageDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Sequence object to train a model. This class allows
    for scrambling the images using a scrambler, which is a 
    a scheme for shuffling the image pixels.
    """
    def __init__(
        self, 
        features,
        labels,
        batch_size=32,
        scrambler_array=None,
        normalize=True   
    ):
        self.features   = np.copy(features)
        self.labels     = np.copy(labels)
        self.batch_size = batch_size
        if(scrambler_array is not None):
            self.scrambler = np.copy(scrambler_array)
        else:
            self.scrambler = None
        assert(self.features.shape[0] == self.labels.shape[0])
        self.dataset_length = self.labels.shape[0]
        if( normalize ):
            ## Normalizing images one by one!
            normalization_array = 1.0 / np.amax(self.features, axis=(1,2))
            self.features = self.features * normalization_array[:,np.newaxis,np.newaxis]

    def __len__(self):
        return self.dataset_length // self.batch_size
        
    def __getitem__(self, idx):
        # Features
        features = np.copy(self.features[self.batch_size*idx:self.batch_size*(idx+1),:])
        if(self.scrambler is not None):
            for n in range(self.batch_size):
                features[n,:,:] = features[n,:,:].flatten()[self.scrambler].reshape(
                    self.features.shape[1:])
        # Labels
        labels = self.labels[self.batch_size*idx:self.batch_size*(idx+1)]
        return (features, labels)

    def on_epoch_end(self):
        ## Shuffle features and labels
        shuffle_array = np.linspace( 
             start=0, 
             stop=self.labels.shape[0],
             num=self.labels.shape[0],
             endpoint=False,
             dtype=np.uint16)
        np.random.shuffle(shuffle_array)
        self.labels = self.labels[shuffle_array]
        self.features = self.features[shuffle_array]