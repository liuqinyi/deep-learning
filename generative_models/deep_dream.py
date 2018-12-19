from keras.applications import inception_v3
from keras import backend as K
import numpy as np
import scipy
from keras.preprocessing import image

# 禁用所有与训练相关的操作
K.set_learning_phase(0)
# 构建不包括全连接层的Inception v3 网络，使用预训练的ImageNet权重来加载模型
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
# 将层的名称映射为一个系数，这个系数定量表示该层激活对你要最大化的loss的贡献大小，层名称硬编码在内置的model中
layer_contributions = {'mixed2': 0.2, 'mixed3': 3.0, 'mixed4': 2., 'mixed5': 1.5}

# 将层的名称映射为层的实例
layer_dict = dict([(layer.name, layer) for layer in model.layers])
# 在定义loss时将层的贡献添加到这个标量变量中
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    # 获取层的输出
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
'''
梯度上升过程
'''
# 保存生成图像
dream = model.input
# 计算（loss相对于DreamImage）的梯度
grads = K.gradients(loss, dream)[0]
# 梯度标准化
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)
# 给定一张输出图像，设置一个keras函数来获取loss和grad
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grads_values = outs[1]
    return loss_value, grads_values


# 运行iterations次梯度上升
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


def resize_img(img, size):
    img = np.copy(img, size)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)


def preprocessed_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


step = 0.01  # 梯度上升的步长
num_octave = 3  # 运行梯度上升的尺度个数
octave_scale = 1.4  # 两个尺寸之间的大小比例
iterations = 20  # 每个尺寸上运行梯度上升的步数
max_loss = 10  # 如果loss > 10则终端梯度上升
base_image_path = 'data/img/hd.jpg'
# img --> numpy.array
img = preprocessed_image(base_image_path)

original_shape = img.shape[1:3]
# 得到一个存储运行梯度上升的不同尺度
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
                   for dim in original_shape])
    successive_shapes.append(shape)
# 列表反转，升序
successive_shapes = successive_shapes[::-1]

# 图像numpy 缩放至最小尺寸
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)  # 图像放大
    # 梯度上升，改变梦境图像
    img = gradient_ascent(img, iterations=iterations, step=step, max_loss=max_loss)
    # 讲原始图像的较小版本方法，他会像素化
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # 在这个尺寸上计算原始图像的高质量版本
    same_size_original = resize_img(original_img, shape)
    # 二者的差别就是再方法过程中丢失的细节
    lost_detail = same_size_original - upscaled_shrunk_original_img

    # 丢失细节重新注入DreamImage
    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='output/deep_dream/dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='output/deep_dream/final_dream.png')
