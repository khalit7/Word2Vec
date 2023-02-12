import sys
sys.path.append("utils/")
import yaml
import utils.dataset as dataset
import utils.helper as helper
import utils.trainer as trainer
from torch.utils.tensorboard import SummaryWriter
import torch


if __name__ == '__main__':
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        
    
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    train_loader,vocab = dataset.get_data_loader_and_vocab(model_name,"train",batch_size=batch_size,shuffle=True,vocab=None)
    val_loader,_ = dataset.get_data_loader_and_vocab(model_name,"train",batch_size=batch_size,shuffle=True,vocab=vocab)
    model = helper.get_model_by_name(model_name,vocab_size = len(vocab))

    number_of_epochs = config["epochs"]
    criterion = helper.get_criterion()
    scheduler = None
    lr = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    device = torch.device("mps" if torch.has_mps else "cpu")
    model_path = f"weights/{model_name}"

    writer = SummaryWriter(f".runs/{model_name}")

    trainer = trainer.Trainer(model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,model_name,writer)
    trainer.train()
