import torch, re
import torch.nn.functional as F
from BPE_tokenizer import tokenize_eng


class PredictionModule():
    def __init__(self, tokenizer_eng, tokenizer_pol, encoder_eng, encoder_pol, model, alpha=0.6, max_seq=51):
        self.model, self.device, self.num_blks = model, model.device, model.decoder.num_blks
        self.alpha, self.max_len, self.vocab_size = alpha, max_seq, tokenizer_pol.vocab_size
        self.encode_eng = lambda snt: encoder_eng.encode_snt(tokenize_eng(snt)) + [2]
        self.bos_init = torch.tensor(tokenizer_pol.vocab['<bos>'], device=self.device).view(1, -1)
        self.rev_pol = {v: k for k, v in encoder_pol.vocab_encoder.items()}
        self.eos_id = tokenizer_pol.vocab['<eos>']

    def snt_encoder_input(self, eng_snt):
        X_enc = torch.tensor(self.encode_eng(eng_snt), device=self.device).view(1, -1)
        return X_enc, torch.tensor(X_enc.size(1), device=self.device).view(-1, 1)

    def translate_snt(self, eng_snt, num_k=5):
        pred_pol = self.predict_ids(*self.snt_encoder_input(eng_snt), num_k)
        return self.out_handler([self.rev_pol[x] for x in pred_pol][:-1])
        
    def out_handler(self, out_split):
        out_proc = " ".join(out_split).capitalize()
        out_proc = re.sub(r"([^_])\s+", r"\1", out_proc).replace('_', '')
        return re.sub(r'\s+([,.!?:])', r'\1', out_proc)

    def predict_ids(self, X_enc, valid_len, num_k):
        with torch.inference_mode():
            self.model.eval()
            
            X_dec = self.model.encoder(X_enc, valid_len)
            dec_state = self.model.decoder.init_state(X_dec, valid_len)
            Y_dec, dec_state = self.model.decoder(self.bos_init, dec_state)
            
            log_probs = F.log_softmax(Y_dec, dim=2)
            vals_1, idxs_1 = torch.topk(F.log_softmax(Y_dec, 2), k=num_k, dim=2)
            
            dec_state = [
                X_dec.expand(num_k, valid_len.item(), -1),
                valid_len.expand(num_k, -1), 
                [dec_state[2][i].expand(num_k, 1, -1) for i in range(self.num_blks)]
            ]
            
            logs_tens, seqs_tens = vals_1.view(num_k, 1), idxs_1.view(num_k, 1)
            logs_list, seqs_list = logs_tens.tolist(), seqs_tens.tolist()
            finished_seqs = []
            
            for _ in range(self.max_len-1):
                
                Y_dec, dec_state = self.model.decoder(seqs_tens[:, -1:], dec_state)
                Y_log = F.log_softmax(Y_dec, dim=2).view(num_k, -1) + logs_tens
                
                vals_2, idxs_2 = torch.topk(Y_log.view(-1), k=2*num_k)
                new_seqs, new_logs, par_idxs = [], [], []
                
                for tup in zip(idxs_2 // self.vocab_size, idxs_2 % self.vocab_size, vals_2):
                    par, tok, val = map(torch.Tensor.item, tup)
                    if tok == self.eos_id:
                        full_seq = seqs_list[par] + [tok]
                        finished_seqs.append((full_seq, val, len(full_seq)))
                    else:
                        new_seqs.append(seqs_list[par] + [tok])
                        new_logs.append([val])
                        par_idxs.append(par)
                        
                    if len(new_seqs) == num_k:
                        break
                        
                if not new_seqs or len(finished_seqs) >= num_k:
                    break
                    
                dec_state[2] = [dec_state[2][i][par_idxs] for i in range(self.num_blks)]
                logs_list, seqs_list = new_logs, new_seqs
                logs_tens = torch.tensor(logs_list, device=self.device)
                seqs_tens = torch.tensor(seqs_list, device=self.device)

        best_seq = max(finished_seqs, key=lambda h: h[1] / (h[2] ** self.alpha))[0]
        return best_seq

        
    