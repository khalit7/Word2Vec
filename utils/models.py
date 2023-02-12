import constants as CONSTANTS

import torch.nn as nn
import torch




class cbow_model(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size,CONSTANTS.embed_size)
        self.lin = nn.Linear(CONSTANTS.embed_size,vocab_size)
        
        
    def forward(self,x):
        '''
        x has dimentions batch_size X 2*CONSTANTS.context_size
        '''
        x = self.embed(x) # get embeddings =========> x has dimentions (batch_size X 2*CONSTANTS.context_size X CONSTATNS.embed_size)
        x = x.mean(dim=1) # average embeddings of context words ==> x has dimentions (batch_size X CONSTATNS.embed_size )
        x = self.lin(x)   # get simillarity with all vocab => x has dimentions (batch_size * vocab_size)
        return x
    
class cbow_model_one_embedding_per_token(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,CONSTANTS.embed_size)
        
    def forward(self,x):
        '''
        x has dimentions (batch_size X 2*CONSTANTS.context_size)
        '''
        x = self.embed(x) # get embeddings =========> x has dimentions (batch_size X 2*CONSTANTS.context_size X CONSTATNS.embed_size)
        x = x.mean(dim=1) # average embeddings of context words ==> x has dimentions (batch_size X CONSTATNS.embed_size )
        
        # get all embeddings
        all_embeddings = self.embed.weight
        
        # calculate simillarity of between tha batch embeddings and all embeddings
        output = torch.matmul(x,all_embeddings.t()) # output has the shape (batch_size X vocab_size)
        
        return output
    
        
class skipgram_model(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size,CONSTANTS.embed_size)
        self.lin = nn.Linear(CONSTANTS.embed_size,vocab_size)
        
        
    def forward(self,x):
        '''
        x has dimentions (batch_size)
        '''
        x = self.embed(x) # get embeddings =========> x has dimentions (batch_size X CONSTATNS.embed_size)
        x = self.lin(x)   # get simillarity with all vocab => x has dimentions batch_size * vocab_size
        return x
        