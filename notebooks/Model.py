import torch
from torch import nn, nn.functional as F



"""
======================================================
=====-------------------Seq2Seq------------------=====
======================================================
"""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr, pad_id, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.pad_id = pad_id
        self.device = device
        self.to(self.device)

    def config_optim(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def ce_loss(self, Y_hat, Y):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y, reduction='none')
    
    def loss(self, Y_hat, Y):
        l = self.ce_loss(Y_hat, Y)
        mask = (Y.reshape(-1) != self.pad_id).type(torch.float32)
        return (l * mask).sum() / mask.sum()
    
    def forward(self, batch):
        enc_X, dec_X, enc_valid_lens = batch
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_X, dec_state)[0]

    def batch_step(self, batch):
        eng_batch, pol_batch, eng_val_lens = batch
        
        eng_batch = eng_batch.to(self.device)
        pol_batch = pol_batch.to(self.device)
        eng_val_lens = eng_val_lens.to(self.device)
        
        dec_input = pol_batch[:, :-1]
        dec_target = pol_batch[:, 1:]
        
        l = self.loss(self((eng_batch, dec_input, eng_val_lens)), dec_target)
        return l



"""
======================================================
=====-------------------Encoder------------------=====
======================================================
"""
class TransformerEncoder(nn.Module):
    def __init__(self, eng_vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        
        self.embedding = nn.Embedding(eng_vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        
        for i in range(num_blks):
            self.blks.add_module("Block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X_enc, enc_valid_lens):
        X_enc = self.pos_encoding(self.embedding(X_enc) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * self.num_blks
        
        for i, blk in enumerate(self.blks):
            X_enc = blk(X_enc, enc_valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X_enc
        


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X_enc, enc_valid_lens):
        Y = self.addnorm1(X_enc, self.attention(X_enc, X_enc, X_enc, enc_valid_lens))
        return self.addnorm2(Y, self.ffn(Y))



"""
======================================================
=====-------------------Decoder------------------=====
======================================================
"""
class TransformerDecoder(nn.Module):
    def __init__(self, pol_vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        
        self.embedding = nn.Embedding(pol_vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.dense = nn.LazyLinear(pol_vocab_size)
        self.blks = nn.Sequential()
        
        for i in range(num_blks):
            self.blks.add_module("Block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X_dec, state):
        X_dec = self.pos_encoding(self.embedding(X_dec) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * self.num_blks for _ in range(2)]
        
        for i, blk in enumerate(self.blks):
            X_dec, state = blk(X_dec, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X_dec), state



class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, id_blk):
        super().__init__()
        self.id_blk = id_blk
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X_dec, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        if state[2][self.id_blk] is None:
            key_values = X_dec
        else:
            key_values = torch.cat((state[2][self.id_blk], X_dec), dim=1)
        state[2][self.id_blk] = key_values

        if self.training:
            batch_size, seq_length, _ = X_dec.shape
            dec_valid_lens = torch.arange(1, seq_length + 1, device=dec_X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X_dec2 = self.attention1(X_dec, key_values, key_values, dec_valid_lens)
        Y_1 = self.addnorm1(X_dec, X_dec2)

        Y_2 = self.attention2(Y_1, enc_outputs, enc_outputs, enc_valid_lens)
        Y_3 = self.addnorm2(Y_1, Y_2)
        return self.addnorm3(Y_3, self.ffn(Y_3)), state



"""
======================================================
=====------------------Attention-----------------=====
======================================================
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.num_heads = num_heads
        
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=use_bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=use_bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            
        out = self.attention(queries, keys, values, valid_lens)
        out_transposed = self.transpose_out(out)
        return self.W_o(out_transposed)
        
    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_out(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2]).permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)



class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.mT) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)



def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_len, value=-1e6):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        X = _sequence_mask(X, valid_lens, value=-1e6)
        return nn.functional.softmax(X, dim=-1)



"""
======================================================
=====------------------Components----------------=====
======================================================
"""
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, seq_length=34):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, seq_length, num_hiddens))
        X = torch.arange(seq_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)  # trzeba zmienic ostatni pewnie na :X.shape[2]
        return self.dropout(X)



class AddNorm(nn.Module):
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(num_hiddens)

    def forward(self, X, Y):
        return self.layernorm(self.dropout(Y) + X)



class PositionWiseFFN(nn.Module): 
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))