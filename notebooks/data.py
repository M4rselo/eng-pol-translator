import pandas as pd
import re

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
    return df_proc
    
def tokenize_snt(snt, regex_del, tok_func):        
    snt = re.sub(regex_del, '', snt.lower())
    snt = re.sub(r'\s*([.,!?])\s*', r' \1 ', snt)
    return tok_func(re.split(r'\s+', snt.strip()))