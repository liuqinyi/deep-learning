import keras
import numpy as np
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.models import Model
import matplotlib.pyplot as plt

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_valid = x_valid.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_valid = x_valid.reshape(x_valid.shape[0], -1)

# add random noise
x_train_nosiy = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
x_valid_nosiy = x_valid + 0.3 * np.random.normal(loc=0, scale=1, size=x_valid.shape)
x_train_nosiy = np.clip(x_train_nosiy, 0., 1.)
x_valid_nosiy = np.clip(x_valid_nosiy, 0, 1.)
print(x_train_nosiy.shape, x_valid_nosiy.shape)

input_img = Input(shape=(28 * 28,))
encoded = Dense(500, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

auto_encoder = Model(inputs=input_img, outputs=decoded)
auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
auto_encoder.summary()
auto_encoder.fit(x_train_nosiy, x_train, epochs=20, batch_size=128, verbose=1, validation_data=(x_valid, x_valid))

# decoded test images
decoded_img = auto_encoder.predict(x_valid_nosiy)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # noisy data
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_valid_nosiy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # predict
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # original
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_valid[i].reshape(28, 28))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
plt.show()
