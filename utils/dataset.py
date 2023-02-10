import constants as CONSTANTS

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from functools import partial

def _get_data_itr(split):
    data_itr = WikiText2(split=split)
    
    return data_itr


def _get_tokenizer():
    
    tokenizer = get_tokenizer("basic_english")
    
    return tokenizer

def _build_vocab(data_itr,tokenizer):
    v = build_vocab_from_iterator(map(tokenizer,data_itr),min_freq=CONSTANTS.min_freq,specials=["<unk>"])
    v.set_default_index(v["<unk>"])
    
    return v


def _CBOW_collate_fn(batch,text_to_idxs_pipeline):
    '''
    takes a batch of paragraphs from the dataset, returns a batch of X,Y
    '''
    X=[]
    Y=[]
    for b in batch:
        transformed_b = text_to_idxs_pipeline(b)
        # if b is too short, skip it
        if len(b) < CONSTANTS.context_size*2 + 1:
            continue
        # if b is too long, truncate it
        if len(b) > CONSTANTS.max_seq_len:
            b = b[0:CONSTANTS.max_seq_len]
            
        for i in range(CONSTANTS.context_size,len(transformed_b)-CONSTANTS.context_size):
            middle_token = transformed_b[i]
            context_tokens = transformed_b[i-CONSTANTS.context_size:i] + transformed_b[i+1:i+1+CONSTANTS.context_size]
            
            X.append(context_tokens)
            Y.append(middle_token)
            
    X = torch.tensor(X,dtype=torch.float32)
    Y = torch.tensor(Y,dtype=torch.float32)
    
    return X,Y


def _SKIPGRAM_collate_fn(batch,text_to_idxs_pipeline):
    '''
    takes a batch of paragraphs from the dataset, returns a batch of X,Y
    '''
    X=[]
    Y=[]
    for b in batch:
        transformed_b = text_to_idxs_pipeline(b)
        # if b is too short, skip it
        if len(b) < CONSTANTS.context_size*2 + 1:
            continue
        # if b is too long, truncate it
        if len(b) > CONSTANTS.max_seq_len:
            b = b[0:CONSTANTS.max_seq_len]
            
        for i in range(CONSTANTS.context_size,len(transformed_b)-CONSTANTS.context_size):
            middle_token = transformed_b[i]
            context_tokens = transformed_b[i-CONSTANTS.context_size:i] + transformed_b[i+1:i+1+CONSTANTS.context_size]
            
            for c in context_tokens:
                X.append(middle_token)
                Y.append(c)
            
    X = torch.tensor(X,dtype=torch.float32)
    Y = torch.tensor(Y,dtype=torch.float32)
    
    return X,Y


def get_data_loader_and_vocab(model_name,data_split,batch_size=1,shuffle=True,vocab=None):
    
    # get data iterator:
    data_itr = _get_data_itr(data_split)
    
    # get tokenizer
    tokenizer = _get_tokenizer()
    
    # build vocab
    if vocab is None:
        v = _build_vocab(data_itr,tokenizer)
        
    text_to_idxs_pipeline = lambda x:v(tokenizer(x))
    if model_name == "cbow":
        collate_fn = _CBOW_collate_fn
    elif model_name == "skipgram":
        collate_fn = _SKIPGRAM_collate_fn
    else:
        raise Exception("unknown model name, model name is expected to be either cbow or skipgram.")
    
    dataloader = DataLoader(data_itr, batch_size=batch_size, shuffle=shuffle,collate_fn= partial(collate_fn,text_to_idxs_pipeline=text_to_idxs_pipeline))
    
    return dataloader,v