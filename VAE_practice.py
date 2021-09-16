# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:04:07 2021

@author: kevin
"""

### Hands-on test with VAE in tensorflow and keras framework
# https://pythonprogramming.net/autoencoders-tutorial/
# https://github.com/bnsreenu/python_for_microscopists/blob/master/178_179_variational_autoencoders_mnist.py
# https://github.com/musikalkemist/generating-sound-with-neural-networks/tree/main/11%20Implementing%20VAE/code

# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/
#generative/ipynb/vae.ipynb#scrollTo=9aJ46lHq-M1j
# https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_tensorflow.py
# https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article

# %% imports
import tensorflow as tf  
from tensorflow import keras
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# %% data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

img_width  = x_train.shape[1]
img_height = x_train.shape[2]
num_channels = 1 #MNIST --> grey scale so 1 channel
input_shape = (img_height, img_width, num_channels)

# %%
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# %% Q(Z|X)  encoding
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon   #re-parametric trick!

latent_dim = 2
encoder_inputs = keras.Input(shape=input_shape)
x = keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(16, activation="relu")(x)
z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# %% P(X|Z)  decoding
#latent_inputs = keras.Input(shape=(latent_dim,))
#x = keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
#x = keras.layers.Reshape((7, 7, 64))(x)
#x = keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#x = keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#decoder_outputs = keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
#decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#decoder.summary()

# %% test with custom decoding
# https://www.tensorflow.org/guide/keras/custom_layers_and_models
class test_custom_decoder(keras.layers.Layer):
    def __init__(self, units=32, z_dim=latent_dim,original_dim=input_shape):
        super(test_custom_decoder, self).__init__()
        
        ### Operation here (linear for an example for now~) ###
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(z_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
        self.dense_output = keras.layers.Dense(np.prod(input_shape), activation="sigmoid")
        self.reshape_output = keras.layers.Reshape(original_dim)

    def call(self, inputs):
        hh = tf.matmul(inputs, self.w) + self.b   ### linear operation~~~
        temp = self.dense_output(hh)
        out = self.reshape_output(temp)
        return out

latent_inputs = keras.Input(shape=(latent_dim,))
decoder_outputs = test_custom_decoder()(latent_inputs) ###constumize here~~~
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# %% VAE class
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# %% training process
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=3, batch_size=128)

# %% Analysis
def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(vae)

# %%
def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train)

