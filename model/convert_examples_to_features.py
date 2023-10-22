from torch.utils.data import TensorDataset,DataLoader,RandomSampler
import torch
import numpy as np
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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


def convert_examples_to_features(ori_lst, mask_lst, label_lst, tokenizer, batch_size, infer = False):
    """Loads a data file into a list of `InputBatch`s."""

    if infer==False:
        total_left_data = ori_lst + ori_lst
        total_right_data = mask_lst + label_lst
        all_data = tokenizer(total_left_data, total_right_data, padding=True, return_tensors="pt")
        # labels_ids = tokenizer(label_lst,padding=True, truncation = True, max_length = max_len, return_tensors="pt")['input_ids']
        input_ids, token_type_ids, attention_mask = all_data.input_ids[:len(ori_lst)], all_data.token_type_ids[:len(
            ori_lst)], all_data.attention_mask[:len(ori_lst)]
        labels_ids = all_data.input_ids[len(ori_lst):]
        # ori_lst_0, ori_lst_1, mask_lst_0, mask_lst_1, label_lst_0, label_lst_1 \
        #     = [_[0] for _ in ori_lst],[_[1] for _ in ori_lst],\
        #       [_[0] for _ in mask_lst],[_[1] for _ in mask_lst], \
        #       [_[0] for _ in label_lst], [_[1] for _ in label_lst]

        # total_left
        # ori_data = tokenizer(ori_lst_0, ori_lst_1, padding=True, return_tensors="pt")
        # mask_data = tokenizer(mask_lst_0, mask_lst_1, padding=True, return_tensors="pt")
        # label_data = tokenizer(mask_lst_0, mask_lst_1, padding=True, return_tensors="pt")
    else:
        total_train_data = [x + ' [SEP] ' + y for x, y in zip(ori_lst, mask_lst)]
        total_test_data = [x + ' [SEP] ' + y for x, y in zip(ori_lst, label_lst)]
        all_data = tokenizer(total_train_data + total_test_data, padding=True, return_tensors="pt")
        input_ids, token_type_ids, attention_mask = all_data.input_ids[:len(ori_lst)], all_data.token_type_ids[:len(ori_lst)], all_data.attention_mask[:len(ori_lst)]
        sep_pos = [[pos for pos, _ in enumerate(text) if _ == tokenizer.sep_token_id] for text in input_ids]

        token_type_ids = torch.tensor([[0] * (_[0] ) +
                          [1] * (_[1]-_[0]) +
                          [0] * (_[2]-_[1] ) +
                          [1] * (_[3]-_[2] ) +
                          [0] * (len(input_ids[0]) - _[3] )
                          for _ in sep_pos],dtype=torch.long)
        labels_ids = all_data.input_ids[len(ori_lst):]

    # for index in np.arange(len(text_lst)):
    #     if len(train_data.input_ids[index]) != len(labels_ids[index]):
    #         print('text_lst[index]',text_lst[index])
    #         print('label_lst[index]',label_lst[index])
    #         print('text_ids[index]', train_data.input_ids[index])
    #         print('labels[index]', labels_ids[index])
    # # label_map = {label: i for i, label in enumerate(label_list)}

    # features = []
    # for (ex_index, example) in enumerate(text_lst):
    #     tokens_a = tokenizer.tokenize(example.text_a)


    train_dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels_ids)
    # train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_data), batch_size=batch_size)
    # label_dataloader = DataLoader(labels_data, sampler=RandomSampler(labels_data), batch_size=batch_size)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader
