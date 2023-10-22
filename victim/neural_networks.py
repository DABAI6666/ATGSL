# coding: utf-8
from __future__ import print_function
from config import config
from keras.layers import *

from keras.models import Sequential,Model
from keras.layers import Embedding, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D, Flatten
from word_level_process import get_tokenizer
import numpy as np
import torch as th
from transformers import BertModel
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaTokenizerFast

import os
import codecs


def get_vocab(file_path, emb_dim):
    global embeddings_index
    embeddings_index = {}
    word_to_idx = {}
    idx_to_emb = []
    f = open(file_path)
    idx_to_emb.append(np.zeros((emb_dim)))
    for index,line in enumerate(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        word_to_idx[word] = index + 1
        idx_to_emb.append(coefs)
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index,word_to_idx, idx_to_emb



def get_embedding_index(file_path):
    global embeddings_index
    embeddings_index = {}
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def get_embedding_matrix(data_path, dataset, num_words, embedding_dims):
    # global num_words, embedding_matrix, word_index
    global embedding_matrix, word_index
    word_index = get_tokenizer(data_path, dataset).word_index
    print('Preparing embedding matrix.')
    # num_words = min(num_words, len(word_index))
    embedding_matrix = np.zeros((num_words + 1, embedding_dims))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def word_cnn(data_path, dataset, glove_path=None):
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.wordCNN_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_cnn model...')
    model = Sequential()
    if glove_path:
        file_path = glove_path + r'/glove.6B.{}d.txt'.format(str(embedding_dims))
        get_embedding_index(file_path)
        get_embedding_matrix(data_path, dataset, num_words, embedding_dims)
        model.add(Embedding(  # Layer 0, Start
            input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
            output_dim=embedding_dims,  # Dimensions to generate
            weights=[embedding_matrix],  # Initialize word weights
            input_length=max_len,
            name="embedding_layer",
            trainable=False))
    else:
        model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # for CNN_2, Xinghao cancel comment
    # model.add(Dense(hidden_dims))
    # model.add(Dropout(0.2))z
    # model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation(activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def get_emb(dataset,embedding_dims,file_path,use_glove=False):

    filters = 250
    kernel_size = 3
    hidden_dims = 250

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    # loss = config.loss[dataset]
    # activation = config.activation[dataset]
    num_words = config.num_words[dataset]

    print('Build word_cnn model...')
    model = Sequential()
    if use_glove:
        # file_path = r'../glove.6B.{}d.txt'.format(str(embedding_dims))
        get_embedding_index(file_path)
        get_embedding_matrix(dataset, num_words, embedding_dims)
        model.add(Embedding(  # Layer 0, Start
            input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
            output_dim=embedding_dims,  # Dimensions to generate
            weights=[embedding_matrix],  # Initialize word weights
            input_length=max_len,
            name="embedding_layer",
            trainable=False))
    else:
        model.add(Embedding(num_words, embedding_dims, input_length=max_len))

    # model.compile(
    #               optimizer='adam',
    #               metrics=['accuracy'])

    return model


def word_cnn_2(dataset):
    # Add one fully connected layer
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.wordCNN_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_cnn model...')
    model = Sequential()

    model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # for CNN_2, Xinghao cancel comment
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation(activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def word_cnn_3(dataset):
    # Replace the Relu with Tanh
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.wordCNN_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_cnn model...')
    model = Sequential()

    model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, kernel_size, padding='valid', activation='tanh', strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))

    model.add(Dense(num_classes))
    model.add(Activation(activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def word_cnn_4(dataset):
    # Add one convolutional layer
    filters = 250
    kernel_size = 3
    hidden_dims = 250

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.wordCNN_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_cnn model...')
    model = Sequential()

    model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation(activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def bd_lstm(data_path,dataset,glove_path=None):
    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.LSTM_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_lstm model...')
    model = Sequential()
    if glove_path:
        file_path = glove_path + r'/glove.6B.{}d.txt'.format(str(embedding_dims))
        get_embedding_index(file_path)
        get_embedding_matrix(data_path, dataset, num_words, embedding_dims)
        model.add(Embedding(  # Layer 0, Start
            input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
            output_dim=embedding_dims,  # Dimensions to generate
            weights=[embedding_matrix],  # Initialize word weights
            input_length=max_len,
            name="embedding_layer",
            trainable=False))
    else:
        model.add(Embedding(num_words, embedding_dims, input_length=max_len))

    # model.add(LSTM(128, name="lstm_layer", dropout=drop_out, recurrent_dropout=drop_out))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation, name="dense_one"))

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model



def lstm(dataset, use_glove=True):
    drop_out = 0.3

    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.LSTM_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_lstm model...')
    model = Sequential()
    if use_glove:
        # file_path = r'../../glove.6B.' + str(embedding_dims) + 'd.txt'
        file_path = r'../glove.6B.{}d.txt'.format(str(embedding_dims))

        get_embedding_index(file_path)
        get_embedding_matrix(dataset, num_words, embedding_dims)
        model.add(Embedding(  # Layer 0, Start
            input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
            output_dim=embedding_dims,  # Dimensions to generate
            weights=[embedding_matrix],  # Initialize word weights
            input_length=max_len,
            name="embedding_layer",
            trainable=False))
    else:
        model.add(Embedding(num_words, embedding_dims, input_length=max_len))

    # model.add(LSTM(128, name="lstm_layer", dropout=drop_out, recurrent_dropout=drop_out))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation, name="dense_one"))

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    return model



def word_bert(dataset, hidden_dim):
    loss = config.loss[dataset]
    num_classes = config.num_classes[dataset]
    model = Sequential()
    model.add(Dense(num_classes, activation='sigmoid', input_dim=hidden_dim))
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


class word_bert_model(th.nn.Module):
    def __init__(self, dataset, model_path, device):
        super(word_bert_model, self).__init__()
        self.num_classes = config.num_classes[dataset]
        self.max_length = config.word_max_len[dataset]
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels = self.num_classes).to(device)
        # print(os.getcwd(),'os.cwd()')
        self.tokenizer = BertTokenizer.from_pretrained(model_path,max_length=config.word_max_len[dataset],truncation = True,map_location=device)
        self.device = device

    def forward(self,x):
        encoded_input = self.tokenizer(x, padding=True, max_length=self.max_length, truncation=True)
        tokens_tensor = th.tensor(encoded_input['input_ids']).to(self.device)
        mask_tensor = th.tensor(encoded_input['attention_mask']).to(self.device)
        segmets_tensors = th.tensor(encoded_input['token_type_ids']).to(self.device)
        logits = self.model(tokens_tensor,mask_tensor,segmets_tensors)
        return logits.detach().cpu().numpy()


class word_roberta_model(th.nn.Module):
    def __init__(self, dataset, model_path, device):
        super(word_roberta_model, self).__init__()
        self.num_classes = config.num_classes[dataset]
        self.max_length = config.word_max_len[dataset]
        # self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels = self.num_classes).to(device)
        # # print(os.getcwd(),'os.cwd()')
        # self.tokenizer = BertTokenizer.from_pretrained(model_path,max_length=config.word_max_len[dataset],truncation = True,map_location=device)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=self.num_classes)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, max_length=config.word_max_len[dataset], truncation = True,map_location=device)
        self.device = device

    def forward(self,x):
        encoded_input = self.tokenizer(x, padding=True, max_length=self.max_length, truncation=True)
        tokens_tensor = th.tensor(encoded_input['input_ids']).to(self.device)
        mask_tensor = th.tensor(encoded_input['attention_mask']).to(self.device)
        # segmets_tensors = th.tensor(encoded_input['token_type_ids']).to(self.device)
        logits = self.model(input_ids=tokens_tensor, attention_mask=mask_tensor).logits

        # logits = self.model(tokens_tensor,mask_tensor,segmets_tensors)
        return logits.detach().cpu().numpy()



