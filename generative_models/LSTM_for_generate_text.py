# encoding=utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPooling1D, Dropout
from keras.utils import np_utils
import nltk
from gensim import corpora
from pyecharts import WordCloud

filename = "./data/Alice.txt"
batch_size = 128
epochs = 200
model_json_file = './output/simple_model.json'
model_hd5_file = './output/simple_model.h5'
dict_file = './output/dict_file.txt'
dict_len = 2789
max_len = 20
document_max_len = 33200
document_split = ['.', ',', '?', '!', ';']


def clear_data(str):
    value = str.replace('\ufeff', '').replace('\n', '')
    return value


def load_dataset():
    # 读入文件
    with open(file=filename, mode='r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            # 删除非内容字符
            value = clear_data(line)
            if value != '':
                # 对一行文本进行分词
                for str in nltk.word_tokenize(value):
                    # 跳过章节标题
                    if str == 'CHAPTER':
                        break
                    else:
                        document.append(str.lower())
    return document


def word_to_integer(document):
    # 生成字典
    dic = corpora.Dictionary([document])
    # 保存字典到文本文件
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    for word in document:
        values.append(dic_set[word])
    return values


# 生成词云
def show_word_cloud(document):
    # 需要清除的标点符号
    left_words = ['.', ',', '?', '!', ';', ':', '\'', '(', ')',
                  '[', ']', '’', '‘', '--', '“', '”']
    # 生成字典
    dic = corpora.Dictionary([document])
    # 计算得到每个单词的使用频率
    words_set = dic.doc2bow(document)

    # 生成单词列表和使用频率列表
    words, frequencies = [], []
    for item in words_set:
        key = item[0]
        frequency = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            words.append(word)
            frequencies.append(frequency)
    # 使用pycharts生成词云
    word_cloud = WordCloud(width=1000, height=620)
    word_cloud.add('Alice\'s word cloud', attr=words, value=frequencies, shape='circle', word_size_range=[20, 100])
    word_cloud.render("./output/render.html")


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=dict_len, output_dim=32, input_length=max_len))
    model.add(Conv1D(filters =32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=256))
    model.add(Dropout(.2))
    model.add(Dense(units=dict_len, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def make_dataset(document):
    dataset = np.array(document[0:document_max_len])
    dataset = dataset.reshape(int(document_max_len / max_len), max_len)
    return dataset


def make_y(document):
    dataset = make_dataset(document)
    y = dataset[1:dataset.shape[0], 0]
    return y


def make_x(document):
    dataset = make_dataset(document)
    x = dataset[0:dataset.shape[0] - 1, :]
    return x


if __name__ == '__main__':
    document = load_dataset()
    show_word_cloud(document)
    # 将单词转换为整数
    values = word_to_integer(document)
    x_train = make_x(values)
    y_train = make_y(values)
    # one-hot编码
    y_train = np_utils.to_categorical(y_train, dict_len)
    model = build_model()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    # 保存模型至Json文件
    model_json = model.to_json()
    with open(model_json_file, 'w') as file:
        file.write(model_json)
    # 保存权重至hd5文件
    model.save_weights(model_hd5_file)
