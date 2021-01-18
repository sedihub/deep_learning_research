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


class GANModel(tf.keras.Model):
    """A class based on `tf.keras.Model` with modified `train_step`.
    """
    def __init__(self, discriminator_model, generator_model, name="gan_model"):
        super(GANModel, self).__init__(name=name)
        self.discriminator_model = discriminator_model
        self.generator_model = generator_model
    
    def call(self, inputs, training=False):
        input_0, input_1, input_2 = inputs
        image = tf.cond(
            tf.math.reduce_any(input_0),
            lambda: input_1,
            lambda: self.generator_model(input_2, training=training))
        # image = self.generator_model(input_2, training=training)
        # image = tf.keras.layers.multiply([image, input_0])
        # image = tf.keras.layers.add([image, input_1])
        return self.discriminator_model(image, training=training)
    
    def train_step(self, data):
        if isinstance(data, tuple):
            if len(data) == 3:
                x, y, sample_weight = data
            elif len(data) == 2:
                x, y = data
                sample_weight = None
            else:
                ValueError(f"\"data\" has the length \"{len(data)}\"! Expect 2 or 3...")
        else:
            raise ValueError(f"Expect a tuple, not \"{type(data)}\"!")
            
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        #
        ## Variables:
        module_variables_dict = {"discriminator": [], "generator": []}
        for layer in self.layers:
            if layer.name not in module_variables_dict:
                raise ValueError(f"\"{layer.name}\" not in {list(module_variables_dict.keys())}!")
            module_variables_dict[layer.name] = layer.trainable_variables
        trainable_variables = []
        trainable_variables.extend(module_variables_dict["discriminator"])
        trainable_variables.extend(module_variables_dict["generator"])
        #
        ## Get gradients:
        gradients_and_vars = []
        gradients = tape.gradient(
            target=loss, sources=trainable_variables, output_gradients=None,
            unconnected_gradients=tf.UnconnectedGradients.ZERO) 
        for idx in range(len(module_variables_dict["discriminator"]), len(trainable_variables)):
            # tf.print(tf.math.reduce_sum(gradients[idx]), end=" ---> ")
            gradients[idx] = -1.0 * gradients[idx]  
            # tf.print(tf.math.reduce_sum(gradients[idx]))
        #
        ## Apply gradients:
        gradients_and_vars = zip(gradients, trainable_variables)
        self.optimizer.apply_gradients(gradients_and_vars)
        #
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
