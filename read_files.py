import os
import re
import sys
import csv,random
csv.field_size_limit(500 * 1024 * 1024)

from config import config
import numpy as np
import string


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_imdb_files(path, filetype):
    """
    filetype: 'train' or 'test'
    """

    all_texts = []
    file_list = []
    # path = r'../../data_set/aclImdb/'
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    all_labels = []
    for _ in range(12500):
        all_labels.append([0, 1])
    for _ in range(12500):
        all_labels.append([1, 0])

    data = list(zip(all_texts,all_labels))
    random.shuffle(data)
    all_texts, all_labels = zip(*data)
    return all_texts, all_labels


def split_imdb_files(path):
    print('Processing IMDB dataset')
    train_texts, train_labels = read_imdb_files(path, 'train')
    test_texts, test_labels = read_imdb_files(path, 'test')
    return train_texts, train_labels, test_texts, test_labels


def read_fake_csv_files_backup(filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    path = r'../../data_set/fake/{}.csv'.format(filetype)
    csv.field_size_limit(500 * 1024 * 1024)

    csvfile = open(path, 'r')

    flag = 0
    for index,line in enumerate(csv.reader(csvfile, delimiter=',', quotechar='"')):
        # print(index,'index',len(line),'len(line)')
        # if index != 5200: continue
        if flag == 0:
            flag = 1
            continue
        if len(line) < 5: continue
        content =  line[3]
        texts.append(content)
        labels_index.append(line[-1])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(config.num_classes['fake'], dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    return texts, labels, labels_index

def read_fake_csv_files(path, filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    # cwd = os.getcwd()
    # print(cwd,'cwd')
    path = path + '{}_tok.csv'.format(filetype)

    csvfile = open(path, 'r')
    content_lst = []
    # w_path = r'./data_set/ag_news_csv/new_{}.csv'.format(filetype)
    # w_csv_file = open(w_path, 'w')
    # writer = csv.writer(w_csv_file,delimiter=',')

    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        content_lst.append(line)
        content = line[0]
        texts.append(content)
        labels_index.append(line[1])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(config.num_classes['fake'], dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    # for line in content_lst:
    #     writer.writerow(line)

    return texts, labels, labels_index

def split_fake_csv_files(path):
    print("Processing Fake CSV dataset")
    train_texts, train_labels, _ = read_fake_csv_files(path,'train')
    # train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    test_texts, test_labels, _ = read_fake_csv_files(path, 'test')
    return train_texts, train_labels, test_texts, test_labels



def read_mr_csv_files(path, file_name):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    content = []
    # file_name = r'../../data_set/mr/{}.txt'.format(file_name)
    # path = '../../data_set/fake/{}_tok.csv'.format(filetype)
    file_name = path + '{}.txt'.format(file_name)
    with open(file_name, 'r', encoding='utf-8') as f:
        # content += [line.strip().split() for line in f if line.strip()]
        content = f.readlines()
        for sentence in content:
            labels_index.append(sentence[0])
            sentence = sentence[2:].strip()
            texts.append(sentence)
            doc_count+=1

    labels = []
    for i in range(doc_count):
        label_class = np.zeros(config.num_classes['mr'], dtype='float32')
        label_class[int(labels_index[i])] = 1
        labels.append(label_class)

    return texts, labels, labels_index




def split_MR_test_files(data_path):
    print("Processing MR Test CSV dataset")
    train_texts, train_labels, _ = read_mr_csv_files(data_path,'train')
    test_texts, test_labels, _ = read_mr_csv_files(data_path, 'test')
    return train_texts, train_labels, test_texts, test_labels





def read_mnli_csv_files(path, file_name, model):

    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    file_name = path + r'mnli_{}.txt'.format(file_name)

    if model == 'bert':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}

    with open(file_name, 'r', encoding='utf-8') as f:
        # content += [line.strip().split() for line in f if line.strip()]
        content = f.readlines()
        for sentence in content:
            sentence_lst = sentence.strip().split('\t')
            labels_index.append(labeldict[sentence_lst[0]])
            sentence = (sentence_lst[1],sentence_lst[2])
            texts.append(sentence)
            doc_count += 1

    labels = []
    for i in range(doc_count):
        label_class = np.zeros(len(labeldict), dtype='float32')
        label_class[int(labels_index[i])] = 1
        labels.append(label_class)

    return texts,  labels

def split_mnli_test_files(path, model):
    'type 0 tokenzier, hownet type 1 can_save type 2 train'
    print("Processing mnli_CSV dataset")
    train_texts, train_labels = read_mnli_csv_files(path, 'train', model)
    test_texts, test_labels = read_mnli_csv_files(path, 'test', model)
    return train_texts, train_labels, test_texts, test_labels


def read_snli_test_csv_files(path, file_name, model):


    # filepath = r'../../data_set/snli_1.0/snli_1.0_test.txt'
    filepath = path + r'snli_1.0_{}.txt'.format(file_name)

    if model == 'bert':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}

    with open(filepath, "r", encoding="utf8") as input_data:
        # ids, premises, hypotheses, labels = [], [], [], []
        texts, labels = [], []

        # Translation tables to remove parentheses and punctuation from
        # strings.
        parentheses_table = str.maketrans({"(": None, ")": None})
        punct_table = str.maketrans({key: " "
                                     for key in string.punctuation})

        # Ignore the headers on the first line of the file.
        next(input_data)

        for line in input_data:
            line = line.strip().split("\t")

            # Ignore sentences that have no gold label.
            if line[0] == "-":
                continue

            premise = line[1]
            hypothesis = line[2]

            # Remove '(' and ')' from the premises and hypotheses.
            premise = premise.translate(parentheses_table)
            hypothesis = hypothesis.translate(parentheses_table)

            premise = premise.translate(punct_table)
            hypothesis = hypothesis.translate(punct_table)


            texts.append((premise,hypothesis))

            label_class = np.zeros(len(labeldict), dtype='float32')
            label_class[int(labeldict[line[0]])] = 1
            labels.append(label_class)




    return texts,  labels



def CleanStr(s):
    s = re.sub(r'([-～【】！、。，？“”()（）.!?''""])', r' ', s)  #
    return s  # 返回的是list





def split_snli_test_files(path, model):
    print("Processing snli Test CSV dataset")
    train_texts, train_labels= read_snli_test_csv_files(path, 'train', model)
    test_texts, test_labels = read_snli_test_csv_files(path, 'test', model)
    return train_texts, train_labels, test_texts, test_labels

#
def read_agnews_files(path, filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    # cwd = os.getcwd()
    # print(cwd,'cwd')
    path = path + '{}.csv'.format(filetype)

    csvfile = open(path, 'r')
    content_lst = []
    # w_path = r'./data_set/ag_news_csv/new_{}.csv'.format(filetype)
    # w_csv_file = open(w_path, 'w')
    # writer = csv.writer(w_csv_file,delimiter=',')

    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        content_lst.append(line)
        content = line[1] + ". " + line[2]
        texts.append(content)
        labels_index.append(line[0])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(config.num_classes['agnews'], dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    # for line in content_lst:
    #     writer.writerow(line)

    return texts, labels, labels_index


def split_agnews_files(data_path):
    print("Processing AG's News dataset")
    train_texts, train_labels, _ = read_agnews_files(data_path,'train')  # 120000
    test_texts, test_labels, _ = read_agnews_files(data_path,'test')  # 7600
    return train_texts, train_labels, test_texts, test_labels



if __name__ == '__main__':
    split_agnews_files()
    # split_weibo_files()
    # split_fake_test_files()
