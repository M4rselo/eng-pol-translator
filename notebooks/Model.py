import torch
from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, lr, pad_id, device):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
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

        