import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import re
from collections import Counter

def upload_tsv(path, col_drop=None, sep='\t', header=None):
    df = pd.read_csv(path, sep=sep, header=header)
    if col_drop:
        df = df.drop(col_drop, axis=1)
    df.columns = ['eng_text', 'pol_text']
    return df

def tokenize_data(df, src_col, tgt_col):
    df_proc = df.copy()
    df_proc[src_col] = df_proc[src_col].apply(
        tokenize_snt, 
        regex_del = r"""['"€;:()$%–—‘’°“”₳+&#=‐…−/£@-]""",
        tok_func = lambda x: x + ['<eos>'])
                                             
    df_proc[tgt_col] = df_proc[tgt_col].apply(
        tokenize_snt, 
        regex_del = r"""[]„”€…";:()'%/’$\u200b\xad–°+₳#'\[=−­`—“@«»-]""",
        tok_func = lambda x: ['<bos>'] + x + ['<eos>'])

    df_proc['eng_len'] = df_proc[src_col].str.len()
    df_proc['pol_len'] = df_proc[tgt_col].str.len()
    return df_proc
    
def tokenize_snt(snt, regex_del, tok_func):        
    snt = re.sub(regex_del, '', snt.lower())
    snt = re.sub(r'\s*([.,!?])\s*', r' \1 ', snt)
    return tok_func(re.split(r'\s+', snt.strip()))

def trim_data(df, src_len, tgt_len, thres):
    df_proc = df.copy()
    return df_proc[(df_proc[src_len] <= thres) & (df_proc[tgt_len] <= thres)]

def set_vocab(df, col, min_freq=2, is_tgt=False):
    leng_c = Counter(df[col].explode())
    leng_c = {k: v for k, v in leng_c.items() if v >= min_freq}
    
    toks_c = dict(sorted(leng_c.items(), key=lambda item: item[1], reverse=True))
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
    if is_tgt:
        vocab['<bos>'] = 3
        
    for tok in vocab:
        toks_c.pop(tok, None)

    for i, k in enumerate(toks_c.keys(), len(vocab)):
        vocab[k] = i
    return vocab

def tokens_to_id(df, src_col, tgt_col, min_freq=2):
    df_proc = df.copy()
    eng_vcb = set_vocab(df_proc, src_col, min_freq=min_freq)
    pol_vcb = set_vocab(df_proc, tgt_col, min_freq=min_freq, is_tgt=True)

    eng_unk, pol_unk = eng_vcb['<unk>'], pol_vcb['<unk>']
    df_proc['src_ids'] = df_proc[src_col].apply(lambda snt: [eng_vcb.get(tok, eng_unk) for tok in snt])
    df_proc['tgt_ids'] = df_proc[tgt_col].apply(lambda snt: [pol_vcb.get(tok, pol_unk) for tok in snt])
    return df_proc, eng_vcb, pol_vcb

def shuffle_split(df, tr_per):
    df = df.sample(frac=1).reset_index(drop=True)
    split_index = int(len(df) * tr_per)
    train_df = df.iloc[:split_index] 
    val_df = df.iloc[split_index:].reset_index(drop=True)
    return train_df, val_df

class EngPolDataset(Dataset):
    def __init__(self, df, src_col, lbl_col):
        self.src_eng = df[src_col]
        self.tgt_pol = df[lbl_col]

    def __len__(self):
        return len(self.src_eng)

    def __getitem__(self, idx):
        return torch.tensor(self.src_eng[idx]), torch.tensor(self.tgt_pol[idx])

def collate_fn(batch):
    eng_batch, pol_batch = zip(*batch)
    eng_padded = pad_sequence(eng_batch, batch_first=True, padding_value=0)
    pol_padded = pad_sequence(pol_batch, batch_first=True, padding_value=0)
    eng_val_lens = (eng_padded != 0).sum(dim=-1, keepdims=True)
    return eng_padded, pol_padded, eng_val_lens

def data_loader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)