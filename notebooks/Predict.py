import torch
import torch.nn.functional as F



class PredictionModule():
    def __init__(self, model, encoder_eng, tokenizer_pol, alpha=0.6, max_len=34):
        self.model, self.device = model, model.device
        self.alpha, self.max_len = alpha, max_len
        self.encode_snt = encoder_eng.encode_snt
        self.rev_pol = {v: k for k, v in encoder_pol.vocab_encoder.items()}
        self.eos_id = tokenizer_pol.vocab['<eos>']
        self.bos_in = torch.tensor(tokenizer_pol.vocab['<bos>']).reshape(1, -1).to(self.device)

    def predict_snt(self, snt_eng, num_k=5):
        snt_ids = seld.encode_snt(snt_eng) + [2]
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
        return [self.rev_pol[t] for t in best[0] if t != self.eos_id], finished

        
# def snt_to_tokens(snt_eng, vocab_eng, device, max_len=34):
#     eng_unk = vocab_eng['<unk>']
#     snt_split = Data.tokenize_snt(snt_eng, r"""['"€;:()$%–—‘’°“”₳+&#=‐…−/£@-]""", lambda x: x + ['<eos>'])[:max_len]
#     snt_ids = [vocab_eng.get(tok, eng_unk) for tok in snt_split]
#     return torch.tensor(snt_ids).reshape(1, -1).to(device), torch.tensor([len(snt_ids)]).reshape(-1, 1).to(device)

# def tokens_to_snt(pred_ids, vcb_pol_rev):
#     snt_pred = list(map(torch.Tensor.item, pred_ids))
#     return [vcb_pol_rev[x] for x in snt_pred]

# def predict_step(snt_eng, vcb_eng, vcb_pol, vcb_pol_rev, model, max_len=34):
#     device = model.device
#     src_eng, src_len = snt_to_tokens(snt_eng, vcb_eng, device, max_len)
#     bos_id, eos_id = vcb_pol['<bos>'], vcb_pol['<eos>']
    
#     with torch.no_grad():
#         model.eval()
#         X_enc = model.encoder(src_eng, src_len)
#         dec_state = model.decoder.init_state(X_enc, src_len)
#         pred_ids = [torch.tensor(bos_id).reshape(1, -1).to(device)]

#         for _ in range(max_len):
#             Y_dec, dec_state = model.decoder(pred_ids[-1], dec_state)
#             pred_ids.append(Y_dec.argmax(2))
#             if pred_ids[-1].item() == eos_id:
#                 break
#     return tokens_to_snt(pred_ids, vcb_pol_rev)
    