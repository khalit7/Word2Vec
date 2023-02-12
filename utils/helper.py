import models
import torch.nn as nn 

def get_model_by_name(model_name,vocab_size):
    
    if model_name == "cbow":
        return models.cbow_model(vocab_size)
    if model_name == "skipgram":
        return models.skipgram_model(vocab_size)
    if model_name == "cbow_one_embed":
        return models.cbow_model_one_embedding_per_token(vocab_size)
    
    else:
        raise Exception("no model has that name")
        
        
def get_criterion():
    return nn.CrossEntropyLoss()