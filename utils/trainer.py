import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2

from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from functools import partial

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self,model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,model_name,writer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.number_of_epochs = number_of_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = criterion
        self.model_name = model_name
        self.model_path = model_path
        self.writer = writer
        
        self.min_val_loss = np.inf
        
        # send model to device
        self.model.to(self.device)
        
    def train(self):
        
        print(f"TRAINING STARTED using device = {self.device} ....",end="\n \n")
        for e in range(1,self.number_of_epochs+1):
            
            print(f"    epoch #{e}")
            
            print(f"        training ...",end=" ")
            epoch_loss = self._train_epoch()
            self.writer.add_scalar("train_loss_per_epoch",epoch_loss,e)
            print("DONE")

            print(f"        evaluating ...",end=" ")
            epoch_loss = self._val_epoch()
            self.writer.add_scalar("val_loss_per_epoch",epoch_loss,e)
            if epoch_loss < self.min_val_loss:
                print(" Saving best model so far ...",end=" ")
                # save model
                self._save_model()
                # update min loss
                self.min_val_loss = epoch_loss
            
            print("DONE")
            
            if self.scheduler:
                self.scheduler.step()

    def _save_model(self):  
        # move state dict to cpu
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict,os.path.join(model_path,"model.pth"))
        
    def _train_epoch(self):
        self.model.train()
        
        running_loss = []
        
        for i,(x,y) in enumerate(self.train_loader,1):
            # send tensors to device
            x,y = x.to(self.device),y.to(self.device)
            
            # get model's predicitions
            pred = self.model(x)
            
            # calculate loss
            loss = self.criterion(pred,y)
            
            # reigister running loss
            step_loss = loss.detach().item()
            self.writer.add_scalar("train_loss_per_step",step_loss,i)
            running_loss.append(step_loss)
            
            # backward prob
            loss.backward()
            self.optimizer.step()
            
            # empty the gradients
            self.optimizer.zero_grad()
        
        epoch_loss = np.mean(running_loss)
        return epoch_loss
    
    def _val_epoch(self):
        self.model.eval()
        
        running_loss = []
        
        for i,(x,y) in enumerate(self.val_loader,1):
            # send tensors to device
            x,y = x.to(self.device),y.to(self.device)
            
            with torch.no_grad():
                # get model's predicitions
                pred = self.model(x)

                # calculate loss
                loss = self.criterion(pred,y)

                # reigister running loss
                step_loss = loss.detach().item()
                self.writer.add_scalar("val_loss_per_step",step_loss,i)
                running_loss.append(step_loss)
        
        epoch_loss = np.mean(running_loss)
        return epoch_loss