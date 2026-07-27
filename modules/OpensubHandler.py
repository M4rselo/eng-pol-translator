import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
from BPE_tokenizer import tokenize_eng, tokenize_pol


def upload_to_dataframe(root, files, num_lines):
    path_eng, path_pol = [root + f for f in files]
    data = defaultdict(list)
    with open(path_eng, encoding='utf_8') as f_eng, open(path_pol, encoding='utf_8') as f_pol:
        for _ in range(num_lines):
            data['eng_text'].append(f_eng.readline().strip())
            data['pol_text'].append(f_pol.readline().strip())
    return pd.DataFrame(data)


class OpenSubDataset():
    def __init__(self, root, files, num_lines):
        self.df = upload_to_dataframe(root, files, num_lines)

    def _mask_distinct(self, df, col1, col2, chars):
        chars_escaped = ''.join(re.escape(c) for c in chars)
        templ = f'[{chars_escaped}]'
        m1 = df[col1].str.contains(templ, regex=True, na=False)
        m2 = df[col2].str.contains(templ, regex=True, na=False)
        return m1 ^ m2

    def _del_row_with_char(self, df, chars):
        for char in chars:
            m1 = df['eng_text'].apply(lambda x: char in x)
            m2 = df['pol_text'].apply(lambda x: char in x)
            df = df[~(m1 | m2)].reset_index(drop=True)
        return df

    def clean(self):
        df = self.df.copy()

        pl_chars = set('ąęćłńóśźżĄĘĆŁŃÓŚŹŻ')
        requir_pol = lambda let: not let.isascii() and let not in pl_chars

        mask_eng = df['eng_text'].apply(lambda x: x.isascii())
        df = df[mask_eng].reset_index(drop=True)

        mask_pol = df['pol_text'].apply(lambda snt: any(requir_pol(let) for let in snt))
        df = df[~mask_pol].reset_index(drop=True)

        mask_len = (df['eng_text'].str.len() >= 3) & (df['pol_text'].str.len() >= 3)
        df = df[mask_len].reset_index(drop=True)

        df['eng_text'] = df['eng_text'].apply(lambda snt: re.sub(r'^ *- *', '', snt))
        df['pol_text'] = df['pol_text'].apply(lambda snt: re.sub(r'^ *- *', '', snt))

        mask_broken = (
            df['eng_text'].apply(lambda snt: bool(re.match(r'.* +- +.*', snt))) |
            df['pol_text'].apply(lambda snt: bool(re.match(r'.* +- +.*', snt)))
        )
        df = df[~mask_broken].reset_index(drop=True)

        mask_punct = (
            df['eng_text'].apply(lambda snt: bool(re.search(r'[.?!][^.?!]', snt))) ^
            df['pol_text'].apply(lambda snt: bool(re.search(r'[.?!][^.?!]', snt)))
        )
        df = df[~mask_punct].reset_index(drop=True)

        mask_end = df['eng_text'].str[-1] != df['pol_text'].str[-1]
        df = df[~mask_end].reset_index(drop=True)

        df['len_ratio'] = df['eng_text'].str.split().str.len() / df['pol_text'].str.split().str.len()
        thres_l = df['len_ratio'].quantile(0.05)
        thres_r = df['len_ratio'].quantile(0.95)
        mask_ratio = (df['len_ratio'] >= thres_l) & (df['len_ratio'] <= thres_r)
        df = df[mask_ratio].drop(columns=['len_ratio']).reset_index(drop=True)

        df['eng_text'] = df['eng_text'].apply(lambda snt: re.sub(r'^ *[*].*', '', snt))
        df['pol_text'] = df['pol_text'].apply(lambda snt: re.sub(r'^ *[*].*', '', snt))

        mask_dist = self._mask_distinct(df, 'eng_text', 'pol_text', '[]()/\\*+#$"️')
        df = df[~mask_dist].reset_index(drop=True)

        df['eng_text'] = df['eng_text'].str.replace('`', "'", regex=False)
        df['pol_text'] = df['pol_text'].str.replace('`', "'", regex=False)
        df['eng_text'] = df['eng_text'].str.replace('~', '', regex=False)
        df['pol_text'] = df['pol_text'].str.replace('~', '', regex=False)

        df = self._del_row_with_char(df, '=+*@#;|_}{')

        mask_dash = (
            df['eng_text'].apply(lambda x: bool(re.search(r' +-[^ ]+', x))) |
            df['pol_text'].apply(lambda x: bool(re.search(r' +-[^ ]+', x)))
        )
        df = df[~mask_dash].reset_index(drop=True)

        df['eng_split'] = df['eng_text'].apply(tokenize_eng)
        df['pol_split'] = df['pol_text'].apply(tokenize_pol)

        thres1 = df['eng_split'].str.len().quantile(0.99)
        thres2 = df['pol_split'].str.len().quantile(0.99)
        mask_trim = (df['eng_split'].str.len() > thres1) | (df['pol_split'].str.len() > thres2)
        df = df[~mask_trim].reset_index(drop=True)

        self.df = df
        return df