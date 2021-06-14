import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module): # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.0, max_len=5000): # original option has dropout=0.1 but this seems weird to me 
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # hack stupid idiot rounding problems
        r = d_model % 2
        d_model += r # make even for the pe bounds by 2 so cos doesn't end up creating 1 more column than necessary and getting upset

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # shape: max_len X 1 X d_model

        # get back to actual requested dim after stupid idiot rounding problems
        d_model -= r
        pe = pe[:,:,:d_model]

        self.register_buffer('pe', pe)
    def forward(self, x):
        embedding = self.pe[:x.size(0), :]
        x = x + embedding
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_forwards = d_model
        self.forwards_embedding = nn.Embedding(max_len, self.d_forwards)

    def forward(self, x, real_lengths=None):
        # x shape: seq len X batch size X hidden dim
        # want to get the positional encodings also with shape seq len X batch size
        seq_len = x.shape[0]
        def make_indices_tensor(indices):
            res = torch.LongTensor([indices]).transpose(0,1)
            if next(self.parameters()).is_cuda:  res = res.cuda()
            return res
        x[:,:,:self.d_forwards]+= self.forwards_embedding( make_indices_tensor(list(range(seq_len)))  )
        # figuring out why pytorch broadcasting sorts the dimensions out properly here will break me,
        # but in the meantime it seems to work so idfk
        return self.dropout(x)

class FullEmbedding(nn.Module):
    def __init__(self, d_model, num_tokens, max_len,
                       positional_encoding_type, positional_dropout):
        super(FullEmbedding,self).__init__()

        position_modules = {'sin':PositionalEncoding,
                            'embed':PositionalEmbedding}
        position_module = position_modules[positional_encoding_type]

        positional_encoding = position_module(d_model,positional_dropout,
            max_len=max_len)
        
        word_embedding = nn.Embedding(num_tokens, d_model)


        self.word = word_embedding
        self.pos = positional_encoding
        self.max_len = max_len

    def forward(self,x,real_lengths=None):
        # x: longtensor of padded 'samples', 
        # x.shape = seq_len X batch_size 
        # (each value: an int in 0,1,..,num_tokens-1)
        res = self.word(x)
        # res shape: seq len X batch size X d_model
        return self.pos(res)