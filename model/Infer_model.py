import sys
sys.path.append('/home/lgy/adv_nlp_attack/')
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification
import torch.nn as nn
import torch,pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from model.esim.model import ESIM

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model.InferSent.models import NLINet
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises']+data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        #         print(max_len)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']
        #                     embed[j, i, :] = np.random.normal(size=(emb_dim))

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches



class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32, device=None):
        super(NLI_infer_InferSent, self).__init__()

        #         self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cpu")
        # torch.cuda.set_device(local_rank)

        # Retrieving model parameters from checkpoint.
        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 1,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }

        print("\t* Building model...")
        self.device= device
        self.model = NLINet(config_nli_model).to(self.device)
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(pretrained_file, map_location='cpu'))

        # construct dataset loader
        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        data_batches = self.dataset.transform_text(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.to(self.device), s2_batch.to(self.device)
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        # build word dict
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict['_OOV_']
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        #         # standardize data format
        #         data = defaultdict(list)
        #         for hypothesis in hypotheses:
        #             data['premises'].append(premise)
        #             data['hypotheses'].append(hypothesis)

        # transform data into indices
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, torch.tensor(max_seq_length - 3,dtype=torch.long))
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader


class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32,device=None):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        # self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cuda")
        self.device = device
        checkpoint = torch.load(pretrained_file)
        # Retrieving model parameters from checkpoint.
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes)
                          # device=self.device)

        self.model.load_state_dict(checkpoint['model'])

        # construct dataset loader
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # transform text data into indices and create batches
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)




class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32, device=None):
        super(NLI_infer_BERT, self).__init__()
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).to(self.device)

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLI_infer_roberta(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32, device=None):
        super(NLI_infer_roberta, self).__init__()
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).to(self.device)

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


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


    def softmax(self, x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def accuracy(self, out, labels):
        outputs = np.argmax(labels, axis=1)
        return np.sum(outputs == out)

    def total_acc_class(self, data, labels, batch_size=36):
        classes_prediction = []
        for index in np.arange(int((len(data) + batch_size) / batch_size)):
            if index == 0:
                # classes_prediction = np.array(
                #     self.predict_batch_classes(data[index * batch_size:(index + 1) * batch_size]))
                classes_prediction =  self.predict_batch_classes(data[index * batch_size:(index + 1) * batch_size]).cpu().numpy()
            else:
                classes_prediction = np.concatenate((classes_prediction, self.predict_batch_classes(
                    data[index * batch_size:(index + 1) * batch_size]).cpu().numpy()))
        acc = self.accuracy(classes_prediction, labels)
        return 1.0 * acc / len(data), classes_prediction

    def predict_prob(self, x):
        premises_lst,hypothese_lst = [[w for w in t[0].rstrip().split()] for t in x], [[w for w in t[1].rstrip().split()] for t in x]
        probs = self.softmax(self.model({'premises': premises_lst, 'hypotheses': hypothese_lst})[0].cpu().numpy())
        return probs

    def total_probs(self, x):
        premises_lst,hypothese_lst = [[w for w in t[0].rstrip().split()] for t in x], [[w for w in t[1].rstrip().split()] for t in x]
        probs = self.softmax(self.model({'premises': premises_lst, 'hypotheses': hypothese_lst}).cpu().numpy())
        return probs



    def predict_batch_classes(self, x):
        premises_lst,hypothese_lst = [[w for w in t[0].rstrip().split()] for t in x], [[w for w in t[1].rstrip().split()] for t in x]
        # print('device',self.model.device)
        probs = self.model({'premises': premises_lst, 'hypotheses': hypothese_lst})
        return torch.argmax(probs, axis=1)


    def predict_classes(self, x):
        premises_lst,hypothese_lst = [[w for w in t[0].rstrip().split()] for t in x], [[w for w in t[1].rstrip().split()] for t in x]
        probs = self.model({'premises': premises_lst, 'hypotheses': hypothese_lst})
        return int(torch.argmax(probs, axis=1)[0])



if __name__ == '__main__':
    line= "So after two or three days he called Callgrove and said , Do n't bother coming in I found a guy better than you .\tHe called up Callgrove to give him a raise ."
    premise,hypothese = line.strip().split('\t')
    premise = [w for w in premise.rstrip().split()]
    hypothese = [w for w in hypothese.rstrip().split()]
    premise_lst = [premise,premise]
    hypothese_lst = [hypothese,hypothese]
    # model_path = r'../../runs/mnli/bert'
    #
    # model = NLI_infer_BERT(model_path)
    # predictor = model.text_pred
    # print("Model built!")
    #
    # orig_probs = predictor({'premises': premise_lst, 'hypotheses': hypothese_lst}).squeeze()
    # orig_label = torch.argmax(orig_probs)
    # orig_prob = torch.argmax(orig_probs,axis = 1)
    target_model_path = '../../runs/mnli/esim/esim_4.pth.tar'
    word_embeddings_path = '../../runs/mnli/esim/preprocess/worddict.pkl'

    model = NLI_infer_ESIM(target_model_path,
                           word_embeddings_path,
                           batch_size=10)
    predictor = model.text_pred
    print("Model built!")
    orig_probs = predictor({'premises': premise_lst, 'hypotheses': hypothese_lst}).squeeze()
    orig_label = torch.argmax(orig_probs)
    # orig_prob = torch.argmax(orig_probs,axis = 1)
    # orig_prob = torch.argmax(orig_probs)

    print('orig_label',orig_label,'orig_probs',orig_probs)






