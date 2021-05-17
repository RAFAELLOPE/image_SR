import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DCSRN
from get_data import Data
import numpy as np
import torch
import matplotlib.pylab as plt

class Trainer:
    def __init__(self, optimizer, batch_size, learning_rate, n_epochs, n_channels, vol_shape=(60,60,60)):
        self.n_channels = n_channels
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vol_shape = vol_shape
        self.model = DCSRN(n_chans=n_channels).to(device=device)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.n_epochs = n_epochs
        
        train_dataset = Data(img_shape=vol_shape)
        val_dataset = Data(train=False, img_shape=vol_shape)

        self.train_loader = DataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
        
        self.val_loader = DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True)
        
        self.loss_fn = nn.MSELoss()


    def psnr_loss(self, x, x_hat):
        eps = 0.001
        mse = torch.sqrt(torch.sum((x - x_hat)**2))
        psnr = (torch.max(x))/(mse + eps)
        loss = 1. / (psnr + eps)
        return loss
    
    def save_model(self, tr_loss, vl_loss, path="/model/model.py"):
        checkpoint = dict()
        checkpoint['Training loss'] = tr_loss
        checkpoint['Validation loss'] = vl_loss
        checkpoint['Model'] = self.model.state_dict()
        checkpoint['Optimizer'] = self.optimizer.state_dict()
        checkpoint['N_channels'] = self.n_channels
        torch.save(checkpoint, path)


    def train(self):
        training_err_list = list()
        validation_err_list = list()

        for epoch in range(1, self.n_epochs + 1):
            tr_loss = 0
            vl_loss = 0

            loss = 0
            for x, y in self.train_loader:
                self.model.train()
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y, y_hat)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item()/self.train_loader.batch_size
            
            loss = 0
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y, y_hat)

                vl_loss += loss.item()/self.val_loader.batch_size
            
            print(f"Epoch {epoch}; Training error:{tr_loss/self.n_epochs}; Validation error:{vl_loss/self.n_epochs};")
            training_err_list.append(tr_loss/self.n_epochs)
            validation_err_list.append(vl_loss/self.n_epochs)
            self.save_model(training_err_list, validation_err_list)
        
        return training_err_list, validation_err_list


if __name__ == "__main__":
    optimizer = optim.Adam
    batch_size = 5
    learning_rate = 1e-5
    n_epochs = 100
    n_channels = 24
    
    trainer = Trainer(optimizer=optimizer,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      n_epochs=n_epochs,
                      n_channels=n_channels)
    
    trainer.train()




