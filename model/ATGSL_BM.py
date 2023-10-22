import argparse
import os
import numpy as np

import random, copy, string

from collections import namedtuple
from typing import List, Optional
from OpenAttack.text_process.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.attack_assist.substitute.word import WordSubstitute, get_default_substitute,get_hownet_substitute
from OpenAttack.attack_assist.filter_words.english import ENGLISH_FILTER_WORDS
from OpenAttack.exceptions import WordNotInDictionaryException


import math

from OpenAttack.metric import UniversalSentenceEncoder, Levenshtein, GPT2LM, LanguageTool







class ATGSL_BM(object):
    def __init__(self,
            max_epochs = 25,
            max_iters= 10,
            tokenizer = None,
            substitute = None,
            filter_words = None,
            cooling_rate = 0.001,
            t_init = 0.1,
            C = 0.0001,
            alpha  = 0.05,
            beta = 0.01,
            theta = 0.05,
            max_len= 100,
            model = None,
            infer = False
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
        self.model = model
        self.punctuation = string.punctuation
        self.infer = infer




    def create_adv_with_candidate(self, adv_text, pos, candidate, victim, true_class):
        tmp_adv_text = list(map(lambda x: x[0], self.tokenizer.tokenize(adv_text)))
        tmp_text = [tmp_adv_text[:pos] + [candidate[_]] + tmp_adv_text[pos+1:] for _ in range(len(candidate))]
        tmp_text = [self.tokenizer.detokenize(_) for _ in tmp_text]
        # print('tmp_text',tmp_text)
        new_prob = victim.total_probs(tmp_text)[true_class]
        # print(new_prob,'new_prob',np.argmin(new_prob),'np.argmin(new_prob)')
        return tmp_text[np.argmin(new_prob)], candidate[np.argmin(new_prob)]

    def dis(self, x1, x2):
        x1,x2 = list(map(lambda x: x[0], self.tokenizer.tokenize(x1))), list(map(lambda x: x[0], self.tokenizer.tokenize(x2)))
        # print('x1,x2',x1, x2)
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


        if self.infer:
            orig_class = victim.predict_classes([(sentence[0],ori_text)])
            ori_prob = victim.predict_prob([(sentence[0],ori_text)])[orig_class]
        else:
            orig_class = victim.predict_classes(ori_text)
            ori_prob = victim.predict_prob(ori_text)[orig_class]





        # S_info = namedtuple('S_info', 'ori_text, adv_text, mask_text')

        pos_index = [pos  for pos, _ in enumerate(x_orig) if _ not in self.filter_words and _ not in self.punctuation]

        res = []
        for i in range(self.max_iters):
            adv_text, pos_len = copy.copy(ori_text), min(self.max_epochs, len(pos_index))
            pos_sample = random.sample(pos_index, pos_len)
            T, replace_lst, replace_pos_lst = self.t_init,[], []
            for t in range(1, pos_len + 1):
                # tmp_doc, tmp_candidate = self.create_adv_with_candidate(adv_text, pos_sample[t],neighbours[pos_sample[t]], victim, orig_class)
                mask_doc = [_ if idx not in pos_sample[:t] else '[MASK]' for idx, _ in enumerate(x_orig) ]
                # print('res', [x_orig[_] for _ in pos_sample[:t]])
                tmp_doc = self.tokenizer.detokenize(mask_doc)
                minn_prob, check_doc = 10000.0, tmp_doc

                # tmp_doc = self.tokenizer.do_detokenize(tmp_doc)
                # tmp_candidate = (x_orig[pos_sample[t]], tmp_candidate)
                if self.infer:
                    tmp_doc_lst, replacement_word_lst = self.model.predict(sentence[0] + ' [SEP] ' + ori_text, sentence[0] + ' [SEP] ' + tmp_doc, ori_text)
                    adv_prob = victim.predict_prob([(sentence[0], adv_text)])[orig_class]

                    for tmp_doc, replacement_word in zip(tmp_doc_lst,replacement_word_lst):
                        if len(self.tokenizer.tokenize(tmp_doc)) != len(x_orig):
                            continue

                        predict_class = victim.predict_classes([(sentence[0], tmp_doc)])
                        new_prob = victim.predict_prob([(sentence[0], tmp_doc)])[orig_class]
                        if new_prob < minn_prob:
                            check_doc = tmp_doc
                            minn_prob = new_prob
                        if predict_class != orig_class:
                            ori_word = [_ for idx, _ in enumerate(x_orig) if idx in pos_sample[:t]]
                            replacement_word = [(x, y) for x, y in zip(ori_word, replacement_word) if x != y]
                            # res.append(self.dis(ori_text, tmp_doc))
                            eva_res = [sentence, {'sub_word_num': len(replacement_word),
                                                  'delta': float(ori_prob - new_prob),
                                                  'replacement_word': replacement_word,
                                                  'tmp_doc': tmp_doc if self.infer == 'False' else [
                                                      (sentence[0], tmp_doc)],
                                                  'ori_prob': float(ori_prob),
                                                  'new_prob': float(new_prob),
                                                  'sim_score': self.use.calc_score(ori_text, tmp_doc),
                                                  'flu_score': self.flu.after_attack(ori_text, tmp_doc),
                                                  'gram_score': len(self.gram.language_tool.check(tmp_doc))}]
                            print(eva_res, 'eva_res')
                            res.append(eva_res)
                            return res

                else:

                    tmp_doc_lst, replacement_word_lst = self.model.predict(ori_text, tmp_doc, ori_text)
                    adv_prob = victim.predict_prob(adv_text)[orig_class]

                    for tmp_doc,replacement_word in zip(tmp_doc_lst,replacement_word_lst):
                        if len(self.tokenizer.tokenize(tmp_doc)) != len(x_orig):
                            continue
                        predict_class = victim.predict_classes(tmp_doc)
                        new_prob = victim.predict_prob(tmp_doc)[orig_class]
                        if new_prob < minn_prob:
                            check_doc = tmp_doc
                            minn_prob = new_prob
                        if predict_class != orig_class:
                            ori_word = [_ for idx, _ in enumerate(x_orig) if idx in pos_sample[:t]]
                            replacement_word = [(x, y) for x, y in zip(ori_word, replacement_word) if x != y]
                            # res.append(self.dis(ori_text, tmp_doc))
                            eva_res = [sentence, {'sub_word_num': len(replacement_word),
                                                  'delta': float(ori_prob - new_prob),
                                                  'replacement_word': replacement_word,
                                                  'tmp_doc': tmp_doc if self.infer == 'False' else [
                                                      (sentence[0], tmp_doc)],
                                                  'ori_prob': float(ori_prob),
                                                  'new_prob': float(new_prob),
                                                  'sim_score': self.use.calc_score(ori_text, tmp_doc),
                                                  'flu_score': self.flu.after_attack(ori_text, tmp_doc),
                                                  'gram_score': len(self.gram.language_tool.check(tmp_doc))}]
                            print(eva_res, 'eva_res')
                            res.append(eva_res)
                            return res

                new_prob = minn_prob
                p, r = min(1.0,self.cal_p(new_prob, adv_prob, check_doc, adv_text, ori_text, T)), random.random()
                T = max(self.t_init - self.C * t + T / (1 + t) * (
                        self.use.calc_score(check_doc, ori_text) - self.use.calc_score(adv_text, ori_text)) +
                        self.theta * ((-1)^(p<r)), 0.01) #+ self.theta * (p<r)
                # if adv_prob - new_prob < 0:
                #     print('p', p)

                if adv_prob - new_prob >  0 or  p > r:
                    adv_text = check_doc
                    continue
                else:
                    pos_sample[t-1] = -1

        return res








