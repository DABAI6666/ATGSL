import numpy as np
import torch
from keras import backend as K
import math,copy, string,random
from torch.autograd import Variable
from keras_preprocessing import sequence
from word_level_process import get_tokenizer, text_to_vector_for_all
from transformers import AutoTokenizer, BertForMaskedLM
from OpenAttack.attack_assist.filter_words.english import ENGLISH_FILTER_WORDS
from word_level_process import word_process
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
import torch as th
from OpenAttack.metric import UniversalSentenceEncoder, Levenshtein, GPT2LM, LanguageTool






class Attack_Classifier:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model, model_name, config, data_path, dataset, bz=36):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        self.model = model
        self.config = config
        self.bz = bz

        self.model_name = model_name
        self.dataset = dataset
        if model_name == 'cnn' or model_name == 'lstm':
            self.tokenizer = get_tokenizer(data_path, dataset)
            input_tensor = model.input
            self.input_tensor = input_tensor

    def accuracy(self, out, labels):
        outputs = np.argmax(labels, axis=1)
        return np.sum(outputs == out)

    def softmax(self, x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def batch_acc(self,x, y):
        classes_prediction = []
        for index in np.arange(int((len(x) + self.bz) / self.bz)):
            if index == 0:
                classes_prediction = self.predict_batch_classes(x[index * self.bz:(index + 1) * self.bz])
            else:
                classes_prediction = np.concatenate((classes_prediction, self.predict_batch_classes(
                    x[index * self.bz:(index + 1) * self.bz])))
        acc = self.accuracy(classes_prediction, y)
        return 1.0 * acc / len(x), classes_prediction

    def batch_probs(self,x):
        classes_prediction = []
        for index in np.arange(int((len(x) + self.bz) / self.bz)):
            if index == 0:
                classes_prediction = self.predict_classes(x[index * self.bz:(index + 1) * self.bz])
            else:
                classes_prediction = np.concatenate((classes_prediction, self.predict_classes(
                    x[index * self.bz:(index + 1) * self.bz])))
        return classes_prediction

    def text_to_vector(self, text):
        maxlen = self.config.word_max_len[self.dataset]
        vector = self.tokenizer.texts_to_sequences([text])
        # vector = [_ for_]
        vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
        return vector

    def predict_prob(self, x):
        if self.model_name == 'cnn' or self.model_name == 'lstm' :
            vector = self.text_to_vector(x)
            prob = self.softmax(self.model.predict(vector).squeeze())
        else:
            prob = self.softmax(self.model([x]))[0]
        return prob

    def predict_classes(self, x):
        if self.model_name == 'cnn' or self.model_name == 'lstm' :
            vector = self.text_to_vector(x)
            prediction = self.model.predict(vector)
        else:
            prediction = self.model([x])[0]
        classes = np.argmax(prediction)
        return classes

    def predict_batch_classes(self, x):
        prediction = self.model(list(x))
        classes = np.argmax(prediction, axis=1)
        return classes


    def total_acc_class(self,x, y):
        if self.model_name == 'cnn' or self.model_name == 'lstm' :
            vector = text_to_vector_for_all(x, self.tokenizer,self.dataset)
            prediction = self.model.predict(vector)
            classes = np.argmax(prediction,axis=1)
            return self.model.evaluate(vector, y)[1], classes
        else:
            return self.batch_acc(x, y)

    def total_probs(self,x):
        if self.model_name == 'cnn' or self.model_name == 'lstm' :
            vector = text_to_vector_for_all(x, self.tokenizer,self.dataset)
            probs = self.model.predict(vector)
            return probs
        else:
            return self.model(x)


class BM_model:
    def __init__(self, path,device,infer=False,t = 3, k = 10):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(path, map_location=device)
        self.model = BertForMaskedLM.from_pretrained(path).to(device)
        self.device = device
        self.infer = infer
        self.top_t = t
        self.top_k = k
        self.use = UniversalSentenceEncoder()

    def predict(self, ori_sentence, mask_sentence, ori_text):
        if self.infer:
            sentence = ori_sentence + ' [SEP] ' + mask_sentence
            inputs = self.bert_tokenizer(sentence,padding=True,return_tensors="pt").to(self.device)
            input_ids, token_type_ids, attention_mask = inputs.input_ids, inputs.token_type_ids, inputs.attention_mask
            sep_pos = [pos for pos, _ in enumerate(input_ids[0]) if _ == self.bert_tokenizer.sep_token_id]
            token_type_ids = torch.tensor([0] * (sep_pos[0]) +
                              [1] * (sep_pos[1] - sep_pos[0]) +
                              [0] * (sep_pos[2] - sep_pos[1]) +
                              [1] * (sep_pos[3] - sep_pos[2]) +
                              [0] * (len(input_ids) - sep_pos[3]),dtype=torch.long)
            logits = self.model(input_ids = input_ids,  token_type_ids = token_type_ids , attention_mask = attention_mask)

        else:
            inputs = self.bert_tokenizer(ori_sentence, mask_sentence,padding=True,return_tensors="pt").to(self.device)
            sep_pos = [pos for pos, _ in enumerate(inputs.input_ids[0]) if _ == self.bert_tokenizer.sep_token_id]
            logits = self.model(**inputs)


        # mask_token_index = (inputs.input_ids == self.bert_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        adv_text = np.array(inputs.input_ids[0].cpu())
        # adv_text_id = [_ if _ != self.bert_tokenizer.mask_token_id else logits[0, pos].argmax(axis=-1) for pos, _ in  enumerate(np.array(inputs.input_ids[0]))]
        # res,replacement_word = [],[]
        all_substitutes,all_word_place = [[]],[[]]
        for pos, _ in enumerate(adv_text):
            if pos <= sep_pos[-2]:continue
            # if _ == self.bert_tokenizer.sep_token_id or _ == self.bert_tokenizer.cls_token_id:continue
            if _ == self.bert_tokenizer.mask_token_id:
                # replace_word = self.bert_tokenizer.decode(int(logits.logits[0,pos].argmax(axis=-1)))
                replace_word = torch.topk(logits.logits[0, pos], 10).indices
                replace_word = [int(_) for _ in replace_word]
                replace_word_decode = self.bert_tokenizer.batch_decode(replace_word)
                pos_res = []
                lev_i,word_i = [],[]
                for index, word in enumerate(replace_word_decode):
                    if word not in string.punctuation and word not in ENGLISH_FILTER_WORDS:
                        # res.append(replace_word[index])
                        # replacement_word.append(word)
                        pos_res.append(replace_word[index])
                        for all_sub, word_ls in zip(all_substitutes,all_word_place):
                            all_sub.append(replace_word[index])
                            word_ls.append(word)
                            word_i.append(copy.copy(word_ls))
                            lev_i.append(copy.copy(all_sub))
                            all_sub.pop()
                            word_ls.pop()
                        if len(pos_res) == self.top_t: break
                # print('replace_word',replace_word)
                # sample_pos = random.sample(range(len(lev_i)), min(len(lev_i), 15))
                # all_substitutes = [lev_i[_] for _ in sample_pos]
                # all_word_place = [word_i[_] for _ in sample_pos]
                all_substitutes = lev_i[:20]
                all_word_place = word_i[:20]
            else:
                for all_sub in all_substitutes:
                    all_sub.append(_)

        # return self.bert_tokenizer.decode(res.split(self.bert_tokenizer.sep_token_id))
        sentence_lst = [self.bert_tokenizer.decode(_).split('[SEP]')[-2].split('[PAD]')[-1] for _ in all_substitutes]
        sample_pos = random.sample(range(len(sentence_lst)),min(len(sentence_lst),20))
        sentence_lst = [sentence_lst[_] for _ in sample_pos]
        word_lst = [all_word_place[_] for _ in sample_pos]
        score = [self.use.calc_score(_, ori_text) for _ in sentence_lst]
        _,  word_list = torch.sort(torch.tensor(score), descending=True)

        return [sentence_lst[i] for i in word_list],[word_lst[i] for i in word_list]

class ForwardInferGradWrapper:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        self.model = model

    def predict_prob(self, x):
        premises_lst,hypothese_lst = [[w for w in t[0].rstrip().split()] for t in x], [[w for w in t[1].rstrip().split()] for t in x]
        probs = self.model({'premises': premises_lst, 'hypotheses': hypothese_lst})
        return probs

    def predict_classes(self, x):
        premises_lst,hypothese_lst = [[w for w in t[0].rstrip().split()] for t in x], [[w for w in t[1].rstrip().split()] for t in x]
        probs = self.model({'premises': premises_lst, 'hypotheses': hypothese_lst})
        return torch.argmax(probs, axis=1)





