import keras
from keras import layers
import numpy as np
import os
from keras.preprocessing import image

latent_dim = 32
height, width = 32, 32
channels = 3
'''
生成器
'''
generator_input = keras.Input(shape=(latent_dim, ))
# 将输入转换为大小为16x16的128个通道的特征图
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 上采样32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
# 生成一个大小为32x32的单通道特征图（即CIFAR10图像形状）
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
# 生成器模型实例化，即(latent_dim, )---> (32, 32, 3)
generator = keras.models.Model(generator_input, x)
generator.summary()

'''
判别器
'''
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)
# 将判别模型实例化，它的形状为(32, 32, 3)d的输入转换为一个二进制分类决策(真/加)
discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.trainable = False
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

gan_input = keras.Input(shape=(latent_dim, ))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 6]
x_train = x_train.reshape((x_train.shape[0], ) + (height, width, channels)).astype('float32') / 255.
iteration = 10000
batch_size = 20
save_dir = './output/gan_out/'
start = 0
for step in range(iteration):
    # 在潜在空间采样随机点， 解码为虚假图像
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generate_images = generator.predict(random_latent_vectors)

    # 合并虚假图像和真实图像
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generate_images, real_images])
    # 合并标签区分真实和虚假图像
    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    # 向标签中添加随机噪声
    labels += 0.05 * np.random.random(labels.shape)
    # 训练判别器
    d_loss = discriminator.train_on_batch(combined_images, labels)
    print('discriminator loss:', d_loss)
    # 在潜在空间采样随机点
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # 合并标签，全部是“真实图像”
    misleading_targets = np.zeros((batch_size, 1))
    # 通过gan模型训练生成器
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    print('adversarial loss:', a_loss)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if step % 100 == 0:
        print("<<<<<<<<<<<<<<<<<<<<<<< step = %d >>>>>>>>>>>>>>>>>>>>>>>>>>>" % step)
        gan.save_weights('output/gan.h5')

        img = image.array_to_img(generate_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))
