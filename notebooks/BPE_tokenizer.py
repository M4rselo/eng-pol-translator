from collections import defaultdict
import re


def tokenize_eng(snt):
    snt = snt.lower()
    snt = re.sub("(?<! )'(?! )", r" '", snt)
    snt = re.sub(r"""([\]?:)\[(/\.\-,%\$"!])""", r" \1 ", snt)
    snt = re.sub(r"\s+", " ", snt).strip()
    return re.split(r'\s+', snt.strip())

    

def tokenize_pol(snt):
    snt = snt.lower()
    snt = re.sub(r"""([\]?:)\[(/\.\-,%\$"!])""", r" \1 ", snt)
    snt = re.sub(r"\s+", " ", snt).strip()
    return re.split(r'\s+', snt.strip())

    

class BPETokenizer():
    def __init__(self, vocab_chrs, vocab_pairs, vocab_size, is_tgt):
        self.vocab_chrs = vocab_chrs
        self.vocab_pairs = vocab_pairs
        self.vocab_size = vocab_size
        self.pair_counts = self.pair_counts_init(vocab_pairs)
        self.vocab = self.vocab_init(vocab_chrs, is_tgt)
        self.base_size = len(self.vocab)

    def vocab_init(self, vocab_chrs, is_tgt):
        vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '_': 3}
        if is_tgt:
            vocab['<bos>'] = 4
            
        uniq_toks = {tok for toks, _ in vocab_chrs for tok in toks}
        uniq_toks.remove('_')
        for i, tok in enumerate(uniq_toks, len(vocab)):
            vocab[tok] = i
        return vocab

    def pair_counts_init(self, vocab_pairs):
        pair_counts = defaultdict(int)
        for toks, freq in vocab_pairs:
            for pair in toks:
                pair_counts[pair] += freq
        return pair_counts

    def train_bpe(self):
        for i in tqdm(range(self.base_size, self.vocab_size)):
            pair = self.most_common_pair()
            self.vocab[pair] = i
            self.replace_pairs(pair)

    def replace_pairs(self, pair):
        pair_con = f"{pair[0]}{pair[1]}"
        p_1, p_2 = map(re.escape, pair)
        
        for i in range(len(self.vocab_chrs)):
            if pair in self.vocab_pairs[i][0]:
                new_chr = re.sub(rf"(?<=\s){p_1}\s{p_2}(?=\s)", pair_con, f" {' '.join(self.vocab_chrs[i][0])} ").split()
                self.vocab_chrs[i][0] = new_chr
                self.update_pair_count(list(zip(new_chr, new_chr[1:])), i)
        self.pair_counts.pop(pair)

    def update_pair_count(self, new_pairs, i):
        old_pairs, val = self.vocab_pairs[i]
        for tup in old_pairs:
            self.pair_counts[tup] -= val  
        for tup in new_pairs:
            self.pair_counts[tup] += val
        self.vocab_pairs[i][0] = new_pairs
      
    def most_common_pair(self):
        return max(self.pair_counts, key=self.pair_counts.get)



class BPEEncoder():
    def __init__(self, bpe_vocab, thres_tup=55, is_tgt=False):
        self.vocab_encoder = {"".join(k): v for k, v in bpe_vocab.items()}
        self.vocab_tuple = list(bpe_vocab.keys())[thres_tup:]

        self.tokenize = tokenize_pol if is_tgt else tokenize_eng
        self.char_factor = lambda word: list(word) + ['_']

    def encode_snt(self, snt):
        return [y for x in self.tokenize(snt) for y in self.encode_word(x)]

    def encode_word(self, word):
        word_factor = self.char_factor(word)
        word_id = self.vocab_encoder.get("".join(word_factor), None)
        if word_id:
            yield from [word_id]
        else:
            word_pairs = list(zip(word_factor, word_factor[1:]))
            for pair in self.vocab_tuple:
                if pair in word_pairs:
                    p_1, p_2 = map(re.escape, pair)
                    word_factor = re.sub(rf"(?<=\s){p_1}\s{p_2}(?=\s)", f"{pair[0]}{pair[1]}", f" {' '.join(word_factor)} ").split()
                    word_pairs = list(zip(word_factor, word_factor[1:]))
            yield from [self.vocab_encoder[x] for x in word_factor]