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


class Train_GAN_Modules:
    def __init__(
        self,
        discriminator_model,
        generator_model,
        batch_size,
        loss_func=None,
        optimizer=None,):
        """Helper class for training GAN modules.

        Args:
            discriminator_model: Discriminator model.
            generator_model: Generator model.
            batch_size: Batch size.
            loss_func: Loss function.
            optimizer: Optimizer.
        """
        if loss_func is None:
            self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, 
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                name="cross_entropy")
        else:
            self.loss_func = loss_func
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=0.001, 
                momentum=0.0, nesterov=False, name="SGD")
        else:
            self.optimizer = optimizer
            
        self.discriminator_model = discriminator_model
        self.generator_model = generator_model
        self.batch_size = batch_size
        
        self._accumulated_losses = {
            "Discriminator Loss on Real Image": 0.0,
            "Discriminator Loss on Generated Image": 0.0,
            "Generator Deception Loss": 0.0} 
        self._counter = 0
        
    def _accumulate_losses(self, l1, l2, l3):
        self._counter += 1
        self._accumulated_losses["Discriminator Loss on Real Image"] += l1
        self._accumulated_losses["Discriminator Loss on Generated Image"] += l2
        self._accumulated_losses["Generator Deception Loss"] += l3
    
    def reset_accumulated_losses(self,):
        for key in self._accumulated_losses:
            self._accumulated_losses[key] = 0.0  
        self._counter = 0
        
    def get_accumulated_losses(self):
        for key in self._accumulated_losses:
            self._accumulated_losses[key] /= self._counter
        return self._accumulated_losses
        
    def train_step(
        self,
        discriminator_input, 
        generator_input,):
        """Training step function.

        Args:
            discriminator_input: Input to the discriminator model.
            generator_input: Input to the generator model.

        Returns:
            None
        """
        # Forward pass:
        with tf.GradientTape(persistent=True) as tape:
            disc_real_image_pred = self.discriminator_model(discriminator_input)
            gen_image = self.generator_model(generator_input)
            disc_gen_image_pred = self.discriminator_model(gen_image)
            #
            real_image_disc_loss = self.loss_func(
                    y_true=tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                    y_pred=disc_real_image_pred)
            gen_image_disc_loss = self.loss_func(
                y_true=tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                y_pred=disc_gen_image_pred)
            gen_image_gen_loss = self.loss_func(
                y_true=tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                y_pred=disc_gen_image_pred)

        # Get gradients:
        disc_gradients_1 = tape.gradient(
            target=real_image_disc_loss, 
            sources=self.discriminator_model.trainable_variables, 
            output_gradients=None,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        disc_gradients_2 = tape.gradient(
            target=gen_image_disc_loss, 
            sources=self.discriminator_model.trainable_variables, 
            output_gradients=None,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        disc_gradients = []
        for g_1, g_2 in zip(disc_gradients_1, disc_gradients_2):
            disc_gradients.append(
                0.5 * (g_1 + g_2))
        #
        gen_gradients = tape.gradient(
            target=gen_image_gen_loss, 
            sources=self.generator_model.trainable_variables, 
            output_gradients=None,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        

        # Update variables:
        all_gradients = []
        all_gradients.extend(disc_gradients)
        all_gradients.extend(gen_gradients)
        #
        all_trainable_variables = []
        all_trainable_variables.extend(self.discriminator_model.trainable_variables)
        all_trainable_variables.extend(self.generator_model.trainable_variables)
        #
        self.optimizer.apply_gradients(
            zip(all_gradients, all_trainable_variables))
        
        losses = (
            float(real_image_disc_loss.numpy()), 
            float(gen_image_disc_loss.numpy()), 
            float(gen_image_gen_loss.numpy()))
        self._accumulate_losses(*losses)
        return losses
