import torch, re
import torch.nn.functional as F
from .BPE_tokenizer import tokenize_eng



class PredictionModule():
    def __init__(self, tokenizer_eng, tokenizer_pol, encoder_eng, encoder_pol, model, alpha=0.6, max_seq=36):
        self.model, self.device, self.num_blks = model, model.device, model.decoder.num_blks
        self.alpha, self.max_len, self.vocab_size = alpha, max_seq, tokenizer_pol.vocab_size
        self.encode_eng = encoder_eng.encode_snt
        self.bos_id = tokenizer_pol.vocab['<bos>']
        self.ref_vocab = {'self_f': tokenizer_pol.vocab['<self_f>'],
                          'self_m': tokenizer_pol.vocab['<self_m>'],
                          'self_na': tokenizer_pol.vocab['<self_na>'],
                          'addr_f': tokenizer_pol.vocab['<addr_f>'],
                          'addr_m': tokenizer_pol.vocab['<addr_m>'],
                          'addr_p': tokenizer_pol.vocab['<addr_p>'],
                          'addr_na': tokenizer_pol.vocab['<addr_na>']}
        
        self.blocked_ids = list(self.ref_vocab.values()) + [self.bos_id]
        self.rev_pol = {v: k for k, v in encoder_pol.vocab_encoder.items()}
        self.eos_id = tokenizer_pol.vocab['<eos>']

    def get_enc_input(self, eng_ids):
        X_enc = torch.tensor(eng_ids, device=self.device).view(1, -1)
        return X_enc, torch.tensor(X_enc.size(1), device=self.device).view(-1, 1)

    def translate_ids(self, eng_ids, pol_ids, num_k=5):
        init_seq = torch.tensor(pol_ids[:2], device=self.device).view(1, -1)
        pred_pol = self.predict_ids(*self.get_enc_input(eng_ids), init_seq, num_k)
        return pol_ids[:2] + pred_pol

    def translate_snt(self, eng_snt, ref_context, num_k=5):
        init_seq = torch.tensor([self.ref_vocab.get(tok) for tok in ref_context] + [self.bos_id], device=self.device).view(1, -1)
        pred_pol = self.predict_ids(*self.get_enc_input(self.encode_eng(tokenize_eng(eng_snt))+[2]), init_seq, num_k)
        return self.out_handler([self.rev_pol[x] for x in pred_pol][:-1])

    def out_handler(self, out_split):
        out_proc = " ".join(out_split).capitalize()
        out_proc = re.sub(r"([^_])\s+", r"\1", out_proc).replace('_', '')
        return re.sub(r'\s+([,.!?:])', r'\1', out_proc)

    def predict_ids(self, X_enc, valid_len, init_seq, num_k):
        with torch.inference_mode():
            self.model.eval()

            X_dec = self.model.encoder(X_enc, valid_len)
            dec_state = self.model.decoder.init_state(X_dec, valid_len)
            Y_dec, dec_state = self.model.decoder(init_seq, dec_state)

            cache_len = dec_state[2][0].shape[1]

            Y_log = F.log_softmax(Y_dec[:, -1:, :], dim=2)
            Y_log[..., self.blocked_ids] = float('-inf')
            vals_1, idxs_1 = torch.topk(Y_log, k=num_k, dim=2)

            dec_state = [
                X_dec.expand(num_k, valid_len.item(), -1),
                valid_len.expand(num_k, -1),
                [dec_state[2][i].expand(num_k, cache_len, -1) for i in range(self.num_blks)]
            ]

            logs_tens, seqs_tens = vals_1.view(num_k, 1), idxs_1.view(num_k, 1)
            logs_list, seqs_list = logs_tens.tolist(), seqs_tens.tolist()
            finished_seqs = []

            for _ in range(self.max_len - 2):
                Y_dec, dec_state = self.model.decoder(seqs_tens[:, -1:], dec_state)
                Y_log = F.log_softmax(Y_dec, dim=2).view(num_k, -1)
                Y_log[:, self.blocked_ids] = float('-inf')
                Y_log = Y_log + logs_tens

                vals_2, idxs_2 = torch.topk(Y_log.view(-1), k=2 * num_k)
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

        if not finished_seqs:
            finished_seqs = [(seqs_list[i]+[2], logs_list[i][0], len(seqs_list[i])+1) for i in range(num_k)]
        best_seq = max(finished_seqs, key=lambda h: h[1] / (h[2] ** self.alpha))[0]
        return best_seq





class PredictionModuleRef():
    def __init__(self, tokenizer_eng, tokenizer_pol, encoder_eng, encoder_pol, model, alpha=0.6, max_seq=36):
        self.model, self.device, self.num_blks = model, model.device, model.decoder.num_blks
        self.alpha, self.max_len, self.vocab_size = alpha, max_seq, tokenizer_pol.vocab_size
        self.encode_eng = encoder_eng.encode_snt
        self.bos_id = tokenizer_pol.vocab['<bos>']
        self.self_vocab = {'f': tokenizer_pol.vocab['<self_f>'],
                           'm': tokenizer_pol.vocab['<self_m>'],
                           'na': tokenizer_pol.vocab['<self_na>']}
        self.blocked_ids = list(self.self_vocab.values()) + [self.bos_id]
        self.rev_pol = {v: k for k, v in encoder_pol.vocab_encoder.items()}
        self.eos_id = tokenizer_pol.vocab['<eos>']

    def get_enc_input(self, eng_ids):
        X_enc = torch.tensor(eng_ids, device=self.device).view(1, -1)
        return X_enc, torch.tensor(X_enc.size(1), device=self.device).view(-1, 1)

    def translate_ids(self, eng_ids, pol_ids, num_k=5):
        init_seq = torch.tensor(pol_ids[:2], device=self.device).view(1, -1)
        pred_pol = self.predict_ids(*self.get_enc_input(eng_ids), init_seq, num_k)
        return pol_ids[:2] + pred_pol

    def translate_snt(self, eng_snt, gender='na', num_k=5):
        init_seq = torch.tensor([self.self_vocab[gender], self.bos_id], device=self.device).view(1, -1)
        pred_pol = self.predict_ids(*self.get_enc_input(self.encode_eng(tokenize_eng(eng_snt))+[2]), init_seq, num_k)
        return self.out_handler([self.rev_pol[x] for x in pred_pol][:-1])

    def out_handler(self, out_split):
        out_proc = " ".join(out_split).capitalize()
        out_proc = re.sub(r"([^_])\s+", r"\1", out_proc).replace('_', '')
        return re.sub(r'\s+([,.!?:])', r'\1', out_proc)

    def predict_ids(self, X_enc, valid_len, init_seq, num_k):
        with torch.inference_mode():
            self.model.eval()

            X_dec = self.model.encoder(X_enc, valid_len)
            dec_state = self.model.decoder.init_state(X_dec, valid_len)
            Y_dec, dec_state = self.model.decoder(init_seq, dec_state)

            cache_len = dec_state[2][0].shape[1]

            Y_log = F.log_softmax(Y_dec[:, -1:, :], dim=2)
            Y_log[..., self.blocked_ids] = float('-inf')
            vals_1, idxs_1 = torch.topk(Y_log, k=num_k, dim=2)

            dec_state = [
                X_dec.expand(num_k, valid_len.item(), -1),
                valid_len.expand(num_k, -1),
                [dec_state[2][i].expand(num_k, cache_len, -1) for i in range(self.num_blks)]
            ]

            logs_tens, seqs_tens = vals_1.view(num_k, 1), idxs_1.view(num_k, 1)
            logs_list, seqs_list = logs_tens.tolist(), seqs_tens.tolist()
            finished_seqs = []

            for _ in range(self.max_len - 2):
                Y_dec, dec_state = self.model.decoder(seqs_tens[:, -1:], dec_state)
                Y_log = F.log_softmax(Y_dec, dim=2).view(num_k, -1)
                Y_log[:, self.blocked_ids] = float('-inf')
                Y_log = Y_log + logs_tens

                vals_2, idxs_2 = torch.topk(Y_log.view(-1), k=2 * num_k)
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

        if not finished_seqs:
            finished_seqs = [(seqs_list[i]+[2], logs_list[i][0], len(seqs_list[i])+1) for i in range(num_k)]
        best_seq = max(finished_seqs, key=lambda h: h[1] / (h[2] ** self.alpha))[0]
        return best_seq
