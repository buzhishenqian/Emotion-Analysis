import numpy as np
import re
import os
import pandas as pd
import jieba
import collections
import pickle


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

# def build_word_dataset(step, word_dict, document_max_len):
#     if step == "train":
#         df = pd.read_csv(TRAIN_PATH, names=["class", "content"])
#     else:
#         df = pd.read_csv(TEST_PATH, names=["class", "content"])
#
#     # Shuffle dataframe
#     df = df.sample(frac=1)
#     x = list(map(lambda d: jieba.cut(d,cut_all=False), df["content"]))
#     x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
#     x = list(map(lambda d: d + [word_dict["<eos>"]], x))
#     x = list(map(lambda d: d[:document_max_len], x))
#     x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
#
#     y = list(map(lambda d: d - 1, list(df["class"])))
#
#     return x, y

class InputExample():
    def __init__(self,text,label):
        self.text=text
        self.label=label


class data_name():
    def __init__(self,data_dir,word_dict=None):
        self.data_dir=data_dir
        # self.num_labels=num_labels
        # self.class_num={}
        self.document_max_len = 100
        self.label_id={}

    # def build_word_dict(self):
        if not os.path.exists("word_dict.pickle"):
            train_df = pd.read_csv(os.path.join(self.data_dir,"train.csv"), names=["class", "content"])
            contents = train_df["content"]

            words = list()
            for content in contents:
                for word in jieba.cut(content, cut_all=False):
                    words.append(word)

            word_counter = collections.Counter(words).most_common()
            self.word_dict = dict()
            self.word_dict["<pad>"] = 0
            self.word_dict["<unk>"] = 1
            self.word_dict["<eos>"] = 2
            for word, _ in word_counter:
                self.word_dict[word] = len(self.word_dict)

            with open("word_dict.pickle", "wb") as f:
                pickle.dump(self.word_dict, f)

        else:
            with open("word_dict.pickle", "rb") as f:
                self.word_dict = pickle.load(f)
        self.word_dict_zize = len(self.word_dict)


    def get_train_example(self):
        # class_num = {}
        # df=pd.read_csv(os.path.join(self.data_dir,"train.csv"))
        # # self.df=pd.read_csv("./data/input/test.csv")
        # print(df.head())
        # for index,row in df.iterrows():
        #     if row[0] not in class_num:
        #         class_num[row[0]]=1
        #     else:
        #         class_num[row[0]]=class_num[row[0]]+1
        # print("类别数目",class_num)
        self.statistics_data("train")
        return self.create_examples("train")

    def get_valid_example(self):
        # class_num = {}
        # df=pd.read_csv(os.path.join(self.data_dir,"valid.csv"))
        # # self.df=pd.read_csv("./data/input/test.csv")
        # print(df.head())
        # for index,row in df.iterrows():
        #     if row[0] not in class_num:
        #         class_num[row[0]]=1
        #     else:
        #         class_num[row[0]]=class_num[row[0]]+1
        # print("类别数目",class_num)
        self.statistics_data("valid")
        return self.create_examples("valid")

    def get_valid_text(self):
        self.statistics_data("valid")
        return self.create_text("valid")

    def statistics_data(self,set_type):
        class_num = {}
        df = pd.read_csv(os.path.join(self.data_dir, set_type+".csv"))
        # self.df=pd.read_csv("./data/input/test.csv")
        print(df.head())
        for index, row in df.iterrows():
            if row[0] not in class_num:
                class_num[row[0]] = 1
            else:
                class_num[row[0]] = class_num[row[0]] + 1
        print("类别数目", class_num)


    def create_examples(self,set_type):
        # text_list=[]
        # label_list=[]
        # df = pd.read_csv(os.path.join(self.data_dir, set_type + ".csv"))
        # # word_dict={'我':1,'是':2,'<unk>':3,'<eos>':4,'<pad>':5}
        # for index,row in df.iterrows():
        #     text=row[1]
        #     # x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), text))
        #     # x = list(map(lambda d: d + [word_dict["<eos>"]], x))
        #     # x = list(map(lambda d: d[:10], x))
        #     # x = list(map(lambda d: d + (10 - len(d)) * [word_dict["<pad>"]], x))
        #     label=row[0]
        #     if label not in self.label_id:
        #         self.label_id[label]=len(self.label_id)+1
        #     text_list.append(text)
        #     label_list.append(label)
            # examples.append(InputExample(text=text,label=self.label_id[label]))
        # return text_list,label_list

        if set_type == "train":
            df = pd.read_csv(os.path.join(self.data_dir,"train.csv"), names=["class", "content"])
            df = df.sample(frac=1)
        else:
            df = pd.read_csv(os.path.join(self.data_dir,"valid.csv"), names=["class", "content"])

        # Shuffle dataframe
        # if set_type=="train":
        #     df = df.sample(frac=1)
        x = list(map(lambda d: jieba.cut(d, cut_all=False), df["content"]))
        x = list(map(lambda d: list(map(lambda w: self.word_dict.get(w, self.word_dict["<unk>"]), d)), x))
        x = list(map(lambda d: d + [self.word_dict["<eos>"]], x))
        x = list(map(lambda d: d[:self.document_max_len], x))
        x = list(map(lambda d: d + (self.document_max_len - len(d)) * [self.word_dict["<pad>"]], x))

        y = list(map(lambda d: d+2, list(df["class"])))

        return x, y

    def create_text(self,set_type):
        df=pd.read_csv(os.path.join(self.data_dir,"valid.csv"),names=["class","content"])
        x=list(map(lambda d: d, list(df["content"])))
        y=list(map(lambda d: d, list(df["class"])))
        return x, y

# a=data_name("./data/data/input",100)
#
# text,label=a.get_train_example()
# print(text,label)



def batch_iter(data, batch_size, num_epochs=None, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    # for epoch in range(num_epochs):

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def write_csv():
    pass


# def batch_iter(inputs, outputs, batch_size, num_epochs):
#     inputs = np.array(inputs)
#     outputs = np.array(outputs)
#
#     num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
#     for epoch in range(num_epochs):
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, len(inputs))
#             yield inputs[start_index:end_index], outputs[start_index:end_index]

