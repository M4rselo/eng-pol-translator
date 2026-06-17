import pandas as pd
import re

def upload_tsv(path, col_drop=None, sep='\t', header=None):
    df = pd.read_csv(path, sep=sep, header=header)
    if col_drop:
        df = df.drop(col_drop, axis=1)
    df.columns = ['eng_text', 'pol_text']
    return df

# def tokenize_snt(snt, is_eng=True):
#     if is_eng:
#         cases_del = r"""['"€;:()$%–—‘’°“”₳+&#=‐…−/£@-]"""
#         tok_func = lambda x: x + ['<eos>']
#     else:    
#         cases_del =  r"""[]„”€…";:()'%/’$\u200b\xad–°+₳#'\[=−­`—“@«»-]"""
#         tok_func = lambda x: ['<bos>'] + x + ['<eos>']
        
#     snt = re.sub(cases_del, '', snt.lower())
#     snt = re.sub(r'\s*([.,!?])\s*', r' \1 ', snt)
#     return tok_func(re.split(r'\s+', snt.strip()))

def tokenize_eng(snt):
    snt = re.sub(r"""['"€;:()$%–—‘’°“”₳+&#=‐…−/£@-]""", '', snt.lower())
    snt = re.sub(r'\s*([.,!?])\s*', r' \1 ', snt)
    return re.split(r'\s+', snt.strip()) + ['<eos>']

def tokenize_pol(snt):
    snt = re.sub(r"""[]„”€…";:()'%/’$\u200b\xad–°+₳#'\[=−­`—“@«»-]""", '', snt.lower())
    snt = re.sub(r'\s*([.,!?])\s*', r' \1 ', snt)
    return ['<bos>'] + re.split(r'\s+', snt.strip()) + ['<eos>']