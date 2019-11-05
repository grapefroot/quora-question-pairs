import numpy as np
import pandas as pd
import torch
import os
from tqdm.autonotebook import tqdm, trange
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers.transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from transformers import transformers
from models import SentenceClf



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
    
    if save is not None and os.path.exists(save):
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
        if os.path.exists(save):
            print('File exists. Appending {} to a file name'.format('_1'))
            torch.save(ds, save + '_1')
        else:
            torch.save(ds, save)
    return ds

def prepare_submission(sc, test_dl):
    #for model 1
    ans = []
    sc.clf.eval()
    for q1, m1, q2, m2 in tqdm(test_dl, position=0):
        with torch.no_grad():
            ans.append(sc(q1.cuda(), m1.cuda(), q2.cuda(), m2.cuda()).softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(ans)

def process_test(model, dl_test):
    #for model 2
    preds = []
    model.eval()
    for batch in tqdm(dl_test, desc='Test Progress', position=0):
        with torch.no_grad():
            tup = tuple(item.cuda() for item in batch[1:])
            model_input = dict(zip(['input_ids', 'attention_mask',  'token_type_ids'], tup))
            logits = model(**model_input)[0]

            preds.append(logits.detach().softmax(axis=1)[:, 1].cpu().numpy())
    #  print(len(preds))
    return np.concatenate(preds)

def get_sub_1(df):
    #for evaluation
    #explore this interactively at model1.ipynb
    model_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_weights, do_lower_case=False)
    model=BertModel.from_pretrained(model_weights, output_hidden_states=True).cuda()
    model.eval()
    sc = SentenceClf(model)
    sc.clf.load_state_dict(torch.load('models/clf_head_weight'))
    model.eval()
    test_ds = QuoraSentences(df, tokenizer, train=False)
    test_dl = DataLoader(test_ds, batch_size=100, collate_fn=collate_fn_test)
    res_cpu = prepare_submission(sc, test_dl)
    return res_cpu
    
def get_sub_2(df):
    #explore this interactively at model2.ipynb
    model_weights = 'bert-base-cased-finetuned-mrpc'
    tokenizer = BertTokenizer.from_pretrained(model_weights, do_lower_case=False)
    model=BertForSequenceClassification.from_pretrained(model_weights, output_hidden_states=True).cuda()
    model.load_state_dict(torch.load('./models/checkpoint_iter_24983_2019-11-04 03:37:20.323658')['model_state_dict'])
    model.eval()
    cached_test_ds = cache_ds(df, tokenizer, save='./data/test_ds_CASED_cached_123', train=False)
    test_sampler = torch.utils.data.SequentialSampler(cached_test_ds)
    dl_test = DataLoader(cached_test_ds, batch_size=100, sampler=test_sampler)
    test_predictions = process_test(model, dl_test)
    return test_predictions

def merge_submissions_to_replicate(sub_1_ans, sub_2_ans):
    #https://www.kaggle.com/c/quora-question-pairs/discussion/31179#latest-203203
    a = 0.165 / 0.37
    b = (1 - 0.165) / (1 - 0.37)
    merged = sub_1_ans * 0.25 + sub_2_ans * 0.75
    return a * merged / (a * merged + b * (1 - merged))

def replicate(final_path=None):
    if final_path is None:
        raise Exception('Provide path for submission')
    
    test = pd.read_csv('data/test.csv', index_col='test_id')[:200]
    print('Replicating. Buckle up...')
    
    print('Calculating model 1 outputs...')
    sub_1 = get_sub_1(test.dropna())
    
    print('Calculating model 2 outputs...')
    sub_2 = get_sub_2(test.dropna())
    
    print('Merging...')
    merged = merge_submissions_to_replicate(sub_1, sub_2)
    
    test['is_duplicate'] = 0
    test.loc[test.dropna().index, 'is_duplicate'] = merged
    
    print('Saving to {}'.format(final_path))
    test[['is_duplicate']].to_csv(final_path)
    print('Successfully saved. Enjoy')