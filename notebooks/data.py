import pandas as pd
def upload_tsv(path, col_drop=None, sep='\t', header=None):
    df = pd.read_csv(path, sep=sep, header=header)
    if col_drop:
        df = df.drop(col_drop, axis=1)
    df.columns = ['eng_text', 'pol_text']
    return df