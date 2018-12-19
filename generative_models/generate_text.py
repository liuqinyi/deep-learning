from nltk import word_tokenize
from gensim import corpora
from keras.models import model_from_json
import numpy as np

model_json_file = './output/simple_model.json'
model_hd5_file = './output/simple_model.h5'
dict_file = './output/dict_file.txt'
words = 200
max_len = 20
my_file = './output/predict.txt'


def load_dict():
    # 从文本导入字典
    dic = corpora.Dictionary.load_from_text(dict_file)
    return dic


def load_model():
    # 从json 加载模型
    with open(model_json_file, 'r') as file:
        model_json = file.read()

    # 加载模型
    model = model_from_json(model_json)
    model.load_weights(model_hd5_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def word_to_integer(document):
    dic = load_dict()
    dic_set = dic.token2id
    values = []
    for word in document:
        values.append(dic_set[word])
    return values


def make_dataset(document):
    dataset = np.array(document)
    dataset = dataset.reshape(1, max_len)
    return dataset


def reverse_document(values):
    dic = load_dict()
    dic_set = dic.token2id
    document = ''
    for value in values:
        word = dic.get(value)
        document = document + str(word) + ' '
    return document


if __name__ == '__main__':
    model = load_model()
    start_doc = 'Alice is a little girl, who has a dream to go to visit the land in the time.'
    document = word_tokenize(start_doc.lower())
    values = word_to_integer(document)
    new_document = [] + values

    for i in range(words):
        x = make_dataset(values)
        prediction = model.predict(x, verbose=0)
        prediction = np.argmax(prediction)
        values.append(prediction)
        new_document.append(prediction)
        values = values[1:]

    new_document = reverse_document(new_document)
    with open(my_file, 'w') as file:
        file.write(new_document)
