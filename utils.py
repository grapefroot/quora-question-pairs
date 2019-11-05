import numpy as np
import torch
import os
from tqdm.autonotebook import tqdm, trange
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


def encode_str(string, *args, **kwargs):
    return tokenizer.encode(string, *args, **kwargs)

class QuoraSentences(torch.utils.data.Dataset):
    def __init__(self, df, tk, train=True):
        self.train = train
        self.df = df
        if self.train:
            self.df.dropna(inplace=True, axis=0)
        self.enc = tk.encode
    
    def __getitem__(self, idx):
        q_1, q_2 = self.df.iloc[idx][['question1', 'question2']]
        enc_1 = self.enc(q_1.lower(), add_special_tokens=True, return_tensors='pt').squeeze()        
        enc_2 = self.enc(q_2.lower(), add_special_tokens=True, return_tensors='pt').squeeze()
        if self.train:
            is_dup = self.df.iloc[idx]['is_duplicate']
            return enc_1, enc_2, is_dup
        return enc_1, enc_2
        
    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    #calculate max length
    max1 = max([item[0].size() for item in batch])
    max2 = max([item[1].size() for item in batch])
    
    q1_batch, q1_mask, q2_batch, q2_mask = [], [], [], []
    y = []
    
    for enc_1, enc_2, is_dup in batch:
        padded_1 = enc_1.new_zeros(max1)
        padded_1[:len(enc_1)] = enc_1
        att_mask_1 = enc_1.new_zeros(max1, dtype=torch.float)
        att_mask_1[:len(enc_1)] = 1
        q1_batch.append(padded_1)
        q1_mask.append(att_mask_1)
        
        padded_2 = enc_2.new_zeros(max2)
        padded_2[:len(enc_2)] = enc_2
        att_mask_2 = enc_2.new_zeros(max2, dtype=torch.float)
        att_mask_2[:len(enc_2)] = 1
        q2_batch.append(padded_2)
        q2_mask.append(att_mask_2)
        
        y.append(is_dup)
        
    return torch.stack(q1_batch), torch.stack(q1_mask), torch.stack(q2_batch), torch.stack(q2_mask), torch.tensor(y)

def collate_fn_test(batch):
    #calculate max length
    max1 = max([item[0].size() for item in batch])
    max2 = max([item[1].size() for item in batch])
    
    q1_batch, q1_mask, q2_batch, q2_mask = [], [], [], []
    
    for enc_1, enc_2 in batch:
        padded_1 = enc_1.new_zeros(max1)
        padded_1[:len(enc_1)] = enc_1
        att_mask_1 = enc_1.new_zeros(max1, dtype=torch.float)
        att_mask_1[:len(enc_1)] = 1
        q1_batch.append(padded_1)
        q1_mask.append(att_mask_1)
        
        padded_2 = enc_2.new_zeros(max2)
        padded_2[:len(enc_2)] = enc_2
        att_mask_2 = enc_2.new_zeros(max2, dtype=torch.float)
        att_mask_2[:len(enc_2)] = 1
        q2_batch.append(padded_2)
        q2_mask.append(att_mask_2)
    return torch.stack(q1_batch), torch.stack(q1_mask), torch.stack(q2_batch), torch.stack(q2_mask)

def evaluate(model, dl_val):
    loss = 0
    steps = 0
    preds = None
    labels = None
    for batch in dl_val:
        model.eval()
        with torch.no_grad():
            tup = tuple(item.cuda() for item in batch[3:])
            model_input = dict(zip(['input_ids', 'attention_mask', 'token_type_ids', 'labels'], tup))
            logloss, logits = model(**model_input)[:2]
            loss += logloss.mean().item()
        steps+=1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = model_input['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, model_input['labels'].detach().cpu().numpy(), axis=0)
    
    y_pred = np.argmax(preds, axis=1)
    keys = ['logloss', 'accuracy', 'f1']
    ll = logloss / steps
    accuracy = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    return zip(keys, (ll, accuracy, f1))

def process_test(model, dl_test):
    preds = []
    model.eval()
    for batch in tqdm(dl_test, desc='Test Progress', position=0):
        with torch.no_grad():
            tup = tuple(item.cuda() for item in batch[1:])
            model_input = dict(zip(['input_ids', 'attention_mask',  'token_type_ids'], tup))
            logits = model(**model_input)[0]
            
        if preds is None:
            preds = [logits.detach().cpu().softmax(axis=1).numpy()[:, 1]]
        else:
            preds.append(logits.detach().cpu().softmax(axis=1).numpy()[:, 1])
      #  print(len(preds))
    return preds

def merge_submissions_to_replicate(sub_1_ans, sub_2_ans):
    #https://www.kaggle.com/c/quora-question-pairs/discussion/31179#latest-203203
    a = 0.165 / 0.37
    b = (1 - 0.165) / (1 - 0.37)
    merged = sub_1_ans * 0.25 + sub_2_ans * 0.75
    return a * merged / (a * merged + b * (1 - merged))

def cache_ds(df, tokenizer, save=None, train=True):
    def process(str_1, str_2):
        max_len=128
        inputs = tokenizer.encode_plus(str_1, str_2, max_length=max_len, add_special_tokens=True)
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 0
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        pad_len = max_len - len(input_ids)
        attn_mask = [1] * len(input_ids)
        input_ids += [pad_token_id] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [pad_token_segment_id] * pad_len
        return input_ids, attn_mask, token_type_ids
    
    if save is not None and os.exists(save):
        print('Trying to load from file {}'.format(save))
        return torch.load(save)
    
    id_list = []
    qid1_list = []
    qid2_list = []
    input_id_list = []
    attn_mask_list = []
    token_type_ids_list = []
    
    if train:
        is_duplicate_list = []

        for id_, (qid1, qid2, question1, question2, is_dup) in tqdm(df.iterrows(), desc='Progress', position=0):
            input_ids, attn_mask, token_type_ids = process(question1, question2)
            id_list.append(id_)
            qid1_list.append(qid1)
            qid2_list.append(qid2)
            input_id_list.append(input_ids)
            attn_mask_list.append(attn_mask)
            token_type_ids_list.append(token_type_ids)
            is_duplicate_list.append(is_dup)

        ds = torch.utils.data.TensorDataset(
            torch.tensor(id_list),
            torch.tensor(qid1_list),
            torch.tensor(qid2_list),
            torch.tensor(input_id_list),
            torch.tensor(attn_mask_list),
            torch.tensor(token_type_ids_list),
            torch.tensor(is_duplicate_list),
        )
    
    else:
        for id_, (question1, question2) in tqdm(df.iterrows(), desc='Progress', position=0):
            input_ids, attn_mask, token_type_ids = process(question1, question2)
            id_list.append(id_)
            input_id_list.append(input_ids)
            attn_mask_list.append(attn_mask)
            token_type_ids_list.append(token_type_ids)

        ds = torch.utils.data.TensorDataset(
            torch.tensor(id_list),
            torch.tensor(input_id_list),
            torch.tensor(attn_mask_list),
            torch.tensor(token_type_ids_list),
        )
    
    if save is not None:
        if os.exists(save):
            print('File exists. Appending {} to a file name'.format('_1'))
            torch.save(ds, save + '_1')
        else:
            torch.save(ds, save)
    return ds