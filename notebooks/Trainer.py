import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
from tqdm.notebook import tqdm
from statistics import mean
from Data import data_loader



"""
======================================================
=====-------------------Trainer------------------=====
======================================================
"""
class TrainerModule():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.scaler = torch.cuda.amp.GradScaler()
        self.curr_epoch = 0

    def plotter_init(self, title):
        self.plotter = PlotterModule(title)

    def prepare_data(self, train_data, val_data):
        self.train_dataloader = data_loader(train_data, self.batch_size)
        self.val_dataloader = data_loader(val_data, self.batch_size)

    def fit(self, model, train_data, val_data, num_epoch, save_path):
        self.prepare_data(train_data, val_data)
        self.model = model
        if getattr(self, 'optim', None) is None:
            self.optim = model.config_optim()
        self.num_epochs = self.curr_epoch + num_epoch

        for epoch in range(self.curr_epoch, self.num_epochs):
            self.curr_epoch += 1
            self.fit_epoch(save_path)
            
    def fit_epoch(self, save_path):
        tr_ltab, val_ltab = [], []
        
        self.model.train()
        pbar_train = tqdm(self.train_dataloader, desc=f"Epoch: {self.curr_epoch} [Train]", leave=False)
        
        for batch in pbar_train:
            with torch.autocast(device_type='cuda'):
                loss = self.model.batch_step(batch)
            #loss = self.model.batch_step(batch)     
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            # loss.backward()
            # self.optim.step()

            tr_ltab.append(loss.item())
            pbar_train.set_postfix(loss=f"{loss.item():.4f}")

        self.model.eval()
        pbar_val = tqdm(self.val_dataloader, desc=f"Epoch: {self.curr_epoch} [Val]", leave=False)

        for batch in pbar_val:
            with torch.no_grad():
                loss = self.model.batch_step(batch)
                val_ltab.append(loss.item())
                pbar_val.set_postfix(loss=f"{loss:.2f}")

        self.plotter.plot(mean(tr_ltab), mean(val_ltab))
        self.save_checkpoint(tr_ltab, val_ltab, save_path)

    def save_checkpoint(self, tr_ltab, val_ltab, save_path):
        torch.save({'epoch': self.curr_epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'train_loss': mean(tr_ltab),
            'val_loss': mean(val_ltab),
            'scaler_state': self.scaler.state_dict()}, 
            save_path + f"{self.curr_epoch}.pt")

    def load_checkpoint(self, model, save_path):
        checkpoint = torch.load(save_path)
        self.curr_epoch = checkpoint['epoch']
        self.model = model
        self.model.load_state_dict(checkpoint['model_state'])
        self.optim = model.config_optim()
        self.optim.load_state_dict(checkpoint['optim_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])



"""
======================================================
=====-------------------Plotter------------------=====
======================================================
"""
class PlotterModule():
    def __init__(self, title):
        self.title = title
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0

    def update_stats(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epoch += 1

    def plot(self, train_loss, val_loss):
        self.update_stats(train_loss, val_loss)
        epochs = range(1, self.epoch + 1)

        clear_output(wait=True)
        plt.figure(figsize=(6, 4), constrained_layout=True)
        plt.plot(epochs, self.train_losses, label=f"Train loss: {self.train_losses[-1]:.4f}")
        plt.plot(epochs, self.val_losses, label=f"Val loss: {self.val_losses[-1]:.4f}")
        plt.xlabel("Epoch")
        plt.title(self.title)
        plt.legend()
        plt.show()
