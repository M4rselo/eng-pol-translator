import torch
import Data

def snt_to_tokens(snt_eng, vocab_eng, device, max_len=34):
    eng_unk = vocab_eng['<unk>']
    snt_split = tokenize_snt(snt_eng, r"""['"€;:()$%–—‘’°“”₳+&#=‐…−/£@-]""", lambda x: x + ['<eos>'])[:max_len]
    snt_ids = [vocab_eng.get(tok, eng_unk) for tok in snt_split]
    return torch.tensor(snt_ids).reshape(1, -1).to(device), torch.tensor([len(snt_ids)]).to(device)

def tokens_to_snt(pred_ids, vcb_pol_rev):
    snt_pred = list(map(torch.Tensor.item, pred_ids))
    return [vcb_pol_rev[x] for x in snt_pred]

def predict_step(snt_eng, vcb_eng, vcb_pol, vcb_pol_rev, model, max_len=34):
    device = model.device
    src_eng, src_len = snt_to_tokens(snt_eng, vcb_eng, device, max_len)
    bos_id, eos_id = vcb_pol['<bos>'], vcb_pol['eos']
    
    with torch.no_grad():
        model.eval()
        X_enc = model.encoder(src_eng, src_len)
        dec_state = model.decoder.init_state(X_enc, src_len)
        pred_ids = [torch.tensor(bos_id).reshape(1, -1).to(device)]

        for _ in range(max_len):
            Y_dec, dec_state = model.decoder(pred_ids[-1], dec_state)
            pred_ids.append(Y_dec.argmax(2))
            if pred_ids[-1].item() == eos_id:
                break
    return tokens_to_snt(pred_ids, vcb_pol_rev)
    