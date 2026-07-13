import torch
import torch.nn.functional as F
from BPE_tokenizer import tokenize_eng


class PredictionModule():
    def __init__(self, model, encoder_eng, encoder_pol, tokenizer_pol, alpha=0.6, max_len=34):
        self.model, self.device = model, model.device
        self.alpha, self.max_len = alpha, max_len
        self.encode_snt = encoder_eng.encode_snt
        self.rev_pol = {v: k for k, v in encoder_pol.vocab_encoder.items()}
        self.eos_id = tokenizer_pol.vocab['<eos>']
        self.bos_in = torch.tensor(tokenizer_pol.vocab['<bos>']).reshape(1, -1).to(self.device)

    def predict_snt(self, snt_eng, num_k=5):
        snt_ids = self.encode_snt(tokenize_eng(snt_eng)) + [2]
        X_enc = torch.tensor(snt_ids).reshape(1, -1).to(self.device)
        val_len = torch.tensor([len(snt_ids)]).reshape(-1, 1).to(self.device)
        finished = []

        with torch.no_grad():
            self.model.eval()
            X_dec = self.model.encoder(X_enc, val_len)
            dec_state = self.model.decoder.init_state(X_dec, val_len)
            Y_dec, dec_state = self.model.decoder(self.bos_in, dec_state)
            vals, inds = torch.topk(F.log_softmax(Y_dec, 2), k=num_k, dim=2)
            
            states = [[X_dec, val_len, [c.clone() if c is not None else None for c in dec_state[2]]] for _ in range(num_k)]
            logs = [vals[:, :, i].item() for i in range(num_k)]
            seqs = [[inds[:, :, i].item()] for i in range(num_k)]

            for _ in range(self.max_len-1):
                rows = []
                for i in range(num_k):
                    tok = torch.tensor(seqs[i][-1]).reshape(1, -1).to(self.device)
                    Y_dec, states[i] = self.model.decoder(tok, states[i])
                    Y_log = F.log_softmax(Y_dec, 2).reshape(-1, )
                    rows.append(logs[i] + Y_log)
                scores = torch.stack(rows)
                vocab  = scores.shape[-1]
                
                vals, idxs = torch.topk(scores.view(-1), 2*num_k)
                new_seqs, new_logs, new_states = [], [], []
                for j in range(2*num_k):
                    rodzic = idxs[j].item() // vocab
                    token  = idxs[j].item() % vocab
                    
                    if token == self.eos_id:
                        seq = seqs[rodzic] + [token]
                        finished.append((seq, vals[j].item(), len(seq)))
                    else:
                        new_seqs.append(seqs[rodzic] + [token])
                        new_logs.append(vals[j].item())
                        new_states.append([X_dec, val_len, [c.clone() for c in states[rodzic][2]]])
                        if len(new_seqs) == num_k:
                            break
                            
                if not new_seqs or len(finished) >= num_k:
                    break
                seqs, logs, states = new_seqs, new_logs, new_states
                
        if not finished:
            finished = [(s, l, len(s)) for s, l in zip(seqs, logs)]
        best = max(finished, key=lambda h: h[1] / (h[2] ** self.alpha))
        return [self.rev_pol[t] for t in best[0] if t != self.eos_id]#, finished

        
    