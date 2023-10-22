import argparse
import os
import numpy as np

import random, copy

from collections import namedtuple
from typing import List, Optional
from OpenAttack.text_process.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.attack_assist.substitute.word import WordSubstitute, get_default_substitute,get_hownet_substitute
from OpenAttack.attack_assist.filter_words.english import ENGLISH_FILTER_WORDS
from OpenAttack.exceptions import WordNotInDictionaryException


import math

from OpenAttack.metric import UniversalSentenceEncoder, Levenshtein, GPT2LM, LanguageTool







class ATGSL_SA(object):
    def __init__(self,
            max_epochs: int = 20,
            max_iters : int = 10,
            tokenizer : Optional[Tokenizer] = None,
            substitute : Optional[WordSubstitute] = None,
            filter_words : List[str] = None,
            # H_temperature: int = 1000,
            # L_temperature: int = 100,
            cooling_rate: int = 0.001,
            t_init: float = 0.1,
            C:float = 0.0001,
            alpha:float = 0.05,
            beta:float= 0.01,
            theta:float=0.05,
            max_len: int= 100,
            do_train: bool = False,
            infer: bool = False
        ):
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)

        if substitute is None:
            substitute = get_default_substitute()
        self.substitute = substitute
        self.hownetsubstitute = get_hownet_substitute()

        if tokenizer is None:
            tokenizer = get_default_tokenizer()
        self.tokenizer = tokenizer
        self.max_iters = max_iters
        if filter_words is None:
            filter_words = ENGLISH_FILTER_WORDS
        self.filter_words = set(filter_words)
        # self.H_temperature = H_temperature
        # self.L_temperature = L_temperature
        self.cooling_rate = cooling_rate
        self.max_epochs = max_epochs
        self.t_init = t_init
        self.C = C
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.use = UniversalSentenceEncoder()
        self.editdis = Levenshtein(Tokenizer)
        self.gram = LanguageTool()
        self.flu = GPT2LM()
        self.max_len = max_len
        self.do_train = do_train
        self.infer = infer

    def get_neighbour_num(self, word, pos):
        try:
            return len(self.substitute(word, pos) + self.hownetsubstitute(word,pos))
        except WordNotInDictionaryException:
            return 0

    def get_neighbours(self, word, pos):
        try:
            return list(
                map(
                    lambda x: x[0],
                    filter(
                        lambda x: len(self.tokenizer.tokenize(x[0]))==1,
                        self.substitute(word, pos) + self.hownetsubstitute(word,pos)),
                )
            )
        except WordNotInDictionaryException:
            return []

    def create_adv_with_candidate(self, adv_text, pos, candidate, victim, true_class, premises = None):
        tmp_adv_text = list(map(lambda x: x[0], self.tokenizer.tokenize(adv_text)))
        tmp_text = [tmp_adv_text[:pos] + [candidate[_]] + tmp_adv_text[pos+1:] for _ in range(len(candidate))]
        tmp_text = [self.tokenizer.detokenize(_) for _ in tmp_text]
        # print('tmp_text',tmp_text)
        if self.infer:
            tmp_sentence = [(premises,_) for _ in tmp_text]
            new_prob = victim.total_probs(tmp_sentence)[:,true_class]
        else:
            new_prob = victim.total_probs(tmp_text)[:,true_class]
        # print(new_prob,'new_prob',np.argmin(new_prob),'np.argmin(new_prob)')
        return tmp_text[np.argmin(new_prob)], candidate[np.argmin(new_prob)]

    def dis(self, x1, x2):
        x1,x2 = list(map(lambda x: x[0], self.tokenizer.tokenize(x1))), list(map(lambda x: x[0], self.tokenizer.tokenize(x2)))
        return [(x1[_],x2[_]) for _ in range(len(x1)) if x1[_] != x2[_]]

    def score(self, p, x1, x2):
        # return p + self.alpha* self.editdis.calc_score(x1,x2) + self.beta * (1 - self.use.calc_score(x1,x2))
        return p + self.alpha* len(self.dis(x1,x2)) + self.beta * (1 - self.use.calc_score(x1,x2))

        # return 0.01

    def cal_p(self,new_p, ori_p, tmp_doc, adv_text, ori_text, T):
        return math.exp(-( self.score(new_p, ori_text, tmp_doc) - self.score(ori_p,ori_text, adv_text) )/T)


    def run_sa(self, victim , sentence):
        if self.infer:
            x_orig = sentence[1].lower()
        else:
            x_orig = sentence.lower()

        x_orig = self.tokenizer.tokenize(x_orig)[:self.max_len]
        x_orig = list(map(lambda x: x[0], x_orig))
        ori_text = self.tokenizer.detokenize(x_orig)

        x_orig_tuple = self.tokenizer.tokenize(ori_text)
        x_orig = list(map(lambda x: x[0], x_orig_tuple))
        x_pos = list(map(lambda x: x[1], x_orig_tuple))
        # if len(x_orig) >= 25: return []
        if self.infer:
            orig_class = victim.predict_classes([(sentence[0],ori_text)])
            ori_prob = victim.predict_prob([(sentence[0],ori_text)])[orig_class]
        else:
            orig_class = victim.predict_classes(ori_text)
            ori_prob = victim.predict_prob(ori_text)[orig_class]

        neighbours = [
            self.get_neighbours(word, pos)[:36]
            if word not in self.filter_words
            else []
            for word, pos in zip(x_orig, x_pos)
        ]

        # S_info = namedtuple('S_info', 'ori_text, adv_text, mask_text')

        pos_index = [index for index in range(len(neighbours)) if len(neighbours[index]) > 0]

        if len(pos_index) == 0:
            return []
        res = []

        for i in range(self.max_iters):
            adv_text, pos_len = copy.copy(ori_text), min(self.max_epochs, len(pos_index))
            pos_sample = random.sample(pos_index, pos_len)
            T, replace_lst, replace_pos_lst = self.t_init,[], []
            for t in range(pos_len):
                if self.infer:
                    tmp_doc, replace_w = self.create_adv_with_candidate(adv_text, pos_sample[t],neighbours[pos_sample[t]], victim, orig_class, sentence[0])
                    new_prob = victim.predict_prob([(sentence[0],tmp_doc)])[orig_class]
                    adv_prob = victim.predict_prob([(sentence[0],adv_text)])[orig_class]
                    predict_class = victim.predict_classes([(sentence[0],tmp_doc)])
                else:
                    tmp_doc,replace_w = self.create_adv_with_candidate(adv_text, pos_sample[t],neighbours[pos_sample[t]], victim, orig_class)
                    new_prob = victim.predict_prob(tmp_doc)[orig_class]
                    adv_prob = victim.predict_prob(adv_text)[orig_class]
                    predict_class = victim.predict_classes(tmp_doc)

                p, r = min(1.0,self.cal_p(new_prob, adv_prob, tmp_doc, adv_text, ori_text, T)), random.random()
                T = max(self.t_init - self.C * t + T / (1 + t) * (
                        self.use.calc_score(tmp_doc, ori_text) - self.use.calc_score(adv_text, ori_text)) +
                        self.theta * ((-1)^(p<r)), 0.01) #+ self.theta * (p<r)
                # if adv_prob - new_prob < 0:
                #     print('p', p)

                if adv_prob - new_prob >  0 or  p > r:
                    replace_lst.append((x_orig[pos_sample[t]],replace_w))
                    replace_pos_lst.append(pos_sample[t])
                    if predict_class != orig_class:
                        eva_res = [sentence, {'sub_word_num': len(replace_lst),
                                              'delta': float(ori_prob - new_prob),
                                              'replacement_word': replace_lst,
                                              'tmp_doc': [(sentence[0], tmp_doc)] if self.infer else tmp_doc,
                                              'ori_prob': float(ori_prob),
                                              'new_prob': float(new_prob),
                                              'sim_score': self.use.calc_score(ori_text, tmp_doc),
                                              'flu_score': self.flu.after_attack(ori_text, tmp_doc),
                                              'gram_score': len(self.gram.language_tool.check(tmp_doc))}]
                        print(eva_res)
                        if self.do_train:
                            mask_doc = [_ if idx not in replace_pos_lst else '[MASK]' for idx, _ in enumerate(x_orig)]
                            mask_doc = self.tokenizer.detokenize(mask_doc)
                            if self.infer:
                                # res.append([(sentence[0],ori_text), (sentence[0], tmp_doc), (sentence[0], mask_doc)])
                                res.append([sentence[0] + ' [SEP] ' + ori_text, sentence[0] + ' [SEP] ' + tmp_doc, sentence[0] + ' [SEP] ' + mask_doc])
                                # print('res_train',[sentence[0] + ' [SEP] ' + ori_text, sentence[0] + ' [SEP] ' + tmp_doc, sentence[0] + ' [SEP] ' + mask_doc])
                            else:
                                res.append([ori_text, tmp_doc, mask_doc])
                            break

                        # print('dis', self.editdis.calc_score(ori_text,tmp_doc))
                        else:
                            # print(eva_res)
                            res.append(eva_res)
                            return res
                    adv_text = copy.copy(tmp_doc)
        return res








