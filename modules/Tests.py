import math, pickle
from collections import defaultdict
from statistics import mean, median
from tqdm.auto import tqdm
from . import Model_ref, Trainer, Predict


"""
======================================================
=====-----------------BleuEvaluation-------------=====
======================================================
"""

class BleuEvaluation():
    def __init__(self, checkpoint_path, artifacts_path, hyperparams, n_ref, lr=0.0001, batch_size=64, pad_id=0,
                 alpha=0.6, device='cuda'):
        """
            hyperparams = (num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, max_seq)         
        """
        num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, max_seq = hyperparams
        tokenizer_eng, tokenizer_pol, encoder_eng, encoder_pol = self.load_artifacts(artifacts_path)
        self.n_prefix = n_ref + 1

        encoder = Model_ref.TransformerEncoder(tokenizer_eng.vocab_size, num_hiddens, ffn_num_hiddens,
                                               num_heads, num_blks, dropout, max_seq)
        decoder = Model_ref.TransformerDecoder(tokenizer_pol.vocab_size, num_hiddens, ffn_num_hiddens,
                                               num_heads, num_blks, dropout, max_seq)
        model = Model_ref.Seq2Seq(encoder=encoder, decoder=decoder, lr=lr, pad_id=pad_id, n_ref=n_ref, device=device)

        trainer = Trainer.TrainerModule(batch_size=64)
        trainer.load_checkpoint(model, checkpoint_path)
        self.predicter = Predict.PredictionModule(tokenizer_eng, tokenizer_pol, encoder_eng, encoder_pol,
                                                  model, alpha=0.6, max_seq=max_seq)

    def load_artifacts(self, artifacts_path):
        artifacts = {}
        for name in ('tokenizer_eng', 'tokenizer_pol', 'encoder_eng', 'encoder_pol'):
            with open(f"{artifacts_path}{name}.pkl", 'rb') as f:
                artifacts[name] = pickle.load(f)
        return artifacts['tokenizer_eng'], artifacts['tokenizer_pol'], artifacts['encoder_eng'], artifacts['encoder_pol']

    def get_avg_bleu(self, sample_df, num_k):
        bleu_scores = []
        for ids_eng, ids_pol in tqdm(zip(sample_df['eng_ids'], sample_df['pol_ids'])):
            pred_pol = self.predicter.translate_ids(ids_eng, ids_pol)
            score = self.bleu(list(map(str, ids_pol[self.n_prefix:-1])), list(map(str, pred_pol[self.n_prefix:-1])), num_k)
            bleu_scores.append(score)
        return mean(bleu_scores), median(bleu_scores)

    def bleu(self, lbl_seq, pred_seq, num_k):
        len_label, len_pred = len(lbl_seq), len(pred_seq)
        score = math.exp(min(0, 1 - len_label / len_pred))
        
        for n in range(1, min(num_k, len_pred) + 1):
            num_matches, label_subs = 0, defaultdict(int)
            for i in range(len_label - n + 1):
                label_subs[' '.join(lbl_seq[i: i + n])] += 1
            for i in range(len_pred - n + 1):
                if label_subs[' '.join(pred_seq[i: i + n])] > 0:
                    num_matches += 1
                    label_subs[' '.join(pred_seq[i: i + n])] -= 1
                    
            num_ngrams = len_pred - n + 1
            if num_matches == 0:
                precision = 0.1 / num_ngrams 
            else:
                precision = num_matches / num_ngrams
            score *= math.pow(precision, math.pow(0.5, n))        
        return score

