""" A utility script containing constructors for discriminator and generator
models.
"""

import tensorflow as tf


def construct_discriminator_model():
    """Constructs and returns a discriminator model.
    """
    inputs = tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32)
    #    
    x = tf.keras.layers.Conv2D(
        filters=8, 
        kernel_size=(5, 5), 
        strides=(1, 1), 
        padding="same",
        data_format="channels_last", 
        dilation_rate=(1, 1), 
        groups=1,
        activation=tf.keras.activations.relu,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None, 
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None,
        name="conv_1"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        filters=32, 
        kernel_size=(5, 5), 
        strides=(1, 1), 
        padding="same",
        data_format="channels_last", 
        dilation_rate=(1, 1), 
        groups=1,
        activation=tf.keras.activations.relu,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None, 
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None, 
        bias_constraint=None,
        name="conv_2"
    )(x)
    x = tf.keras.layers.Flatten(name="flatten")(x) 
    outputs = tf.keras.layers.Dense(
        units=2, 
        activation=tf.keras.activations.softmax, 
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        name="output"
    )(x)
    #
    return tf.keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name="discriminator")


def construct_generator_model(
    input_size=10, 
    output_activation="sigmoid",
    with_batchnorm=True):
    """Constructs and returns a generator model.
    
    Args:
        input_size (int): The input size.
        output_activation (str): Either `linear` or `sigmoid`.
        with_batchnorm (bool): If `False`, will not insert BN layers.
    """
    inputs = tf.keras.Input(shape=(input_size,), dtype=tf.float32)
    #    
    x = tf.keras.layers.Dense(
        units=256, 
        activation=None, 
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        name="dense_1"
    )(inputs)   
    x = tf.keras.layers.LeakyReLU(
        alpha=0.2,
        name="leaky_relu_1"
    )(x)
    x = tf.keras.layers.Dense(
        units=256 * 16, 
        activation=None,  
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        name="dense_2"
    )(x)
    x = tf.keras.layers.LeakyReLU(
        alpha=0.2,
        name="leaky_relu_2"
    )(x)
    x = tf.keras.layers.Reshape(
        target_shape=(4, 4, 256),
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=128, 
        kernel_size=(5, 5), 
        strides=(2, 2), 
        padding="valid",
        output_padding=(0, 0), 
        data_format=None, 
        dilation_rate=(1, 1), 
        activation=None, 
        use_bias=False, 
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        name="deconv_1"
    )(x)
    if with_batchnorm:
        x = tf.keras.layers.BatchNormalization(
            name="batch_norm_1"
        )(x)
    x = tf.keras.layers.LeakyReLU(
        alpha=0.2,
        name="leaky_relu_3"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=64, 
        kernel_size=(5, 5), 
        strides=(2, 2), 
        padding="valid",
        output_padding=(0, 0), 
        data_format=None, 
        dilation_rate=(1, 1), 
        activation=None, 
        use_bias=False, 
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        name="deconv_2"
    )(x)
    if with_batchnorm:
        x = tf.keras.layers.BatchNormalization(
            name="batch_norm_2"
        )(x)
    x = tf.keras.layers.LeakyReLU(
        alpha=0.2,
        name="leaky_relu_4"
    )(x)
    x =  tf.keras.layers.Conv2DTranspose(
        filters=1, 
        kernel_size=(4, 4), 
        strides=(1, 1), 
        padding="valid",
        output_padding=(0, 0), 
        data_format=None, 
        dilation_rate=(1, 1), 
        activation=None, 
        use_bias=False, 
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=None,
        bias_regularizer=None, 
        activity_regularizer=None, 
        kernel_constraint=None,
        bias_constraint=None,
        name="deconv_3"
    )(x)
    if with_batchnorm:
        x = tf.keras.layers.BatchNormalization(
            name="batch_norm_3"
        )(x)
    if output_activation.lower() == "linear":
        outputs = tf.keras.layers.Activation(
            lambda x: 255.0 * tf.keras.activations.linear(x),
            name="output"
        )(x)
    elif output_activation.lower() == "sigmoid":
        outputs = tf.keras.layers.Activation(
            lambda x: 255.0 * tf.keras.activations.sigmoid(x),
            name="output"
        )(x)
    else:
        raise ValueError(f"\`{output_activation}\` is not an expected activation name!")
    #
    return tf.keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name="generator")
