import tqdm
from keras.datasets import mnist
import numpy as np
from keras.layers import *
from keras.preprocessing.image import save_img
from keras.utils import plot_model
from keras.models import Sequential
import tensorflow as tf

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

# get zeros
X_train = x_train[y_train==0]

print(X_train.shape, type(X_train))

# create discriminator
discriminator = Sequential()
discriminator.add(Flatten())
discriminator.add(Dense(512, activation="relu"))
discriminator.add(Dense(256, activation="relu"))
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.compile(loss="binary_crossentropy", optimizer="adam")

# create generator
coding_size = 200

generator = Sequential()
generator.add(Dense(200, activation="relu", input_shape=(coding_size,)))
generator.add(Dense(150, activation="relu"))
generator.add(Dense(28*28*1, activation="relu"))
generator.add(Reshape((28, 28, 1)))

# complete GAN
GAN = Sequential([generator, discriminator])
discriminator.trainable = False
GAN.compile(loss="binary_crossentropy", optimizer="adam")

plot_model(model=GAN, show_trainable=True, show_dtype=True, show_shapes=True, to_file="gan.png")

# train
batch_size = 32
my_data = X_train.reshape((len(X_train), 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

# generator and discriminator seperation from complete model
generator, discriminator = GAN.layers

# training loop

epochs = 5
for epoch in range(epochs):
    for x_batch, tq in zip(dataset, tqdm.tqdm(range(len(dataset)))):
        #discriminator training
        noise = tf.random.normal(shape=[batch_size, coding_size])

        # generator create image
        gen_images = generator(noise)

        # comparing
        fake_vs_real = tf.concat([gen_images, tf.dtypes.cast(x_batch, tf.float32)], axis=0)

        y1 = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)

        discriminator.trainable = True

        discriminator.train_on_batch(fake_vs_real, y1)

        # generator training
        noise = tf.random.normal(shape=[batch_size, coding_size])

        y2 = tf.constant([[1.0]]*batch_size)

        discriminator.trainable = False

        GAN.train_on_batch(noise, y2)

generator.save("generator.h5")
