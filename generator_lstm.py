#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:47:18 2019

@author: shuvayan
"""


# The RNN based encoeder decoder model

# The encoder LSTM


class EncoderLSTM(nn.Module):
    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0, bidirectional=True):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.embedding_size = embedding.size()
        self.num_layers, self.h_dim, self.dropout_p, self.bidirectional = num_layers, h_dim, dropout_p, bidirectional 

        # Create embedding and LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding.weight.data.copy_(embedding)
        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        '''Embed text, get initial LSTM hidden state, and encode with LSTM'''
        x = self.dropout(self.embedding(x)) # embedding
        h0 = self.init_hidden(x.size(1)) # initial state of LSTM
        memory_bank, h = self.lstm(x, h0) # encoding
        return memory_bank, h

    def init_hidden(self, batch_size):
        '''Create initial hidden state of zeros: 2-tuple of num_layers x batch size x hidden dim'''
        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        init = torch.zeros(num_layers, batch_size, self.h_dim)
        init = init.cuda() if use_gpu else init
        h0 = (init, init.clone())
        return h0



class Attention(nn.Module):
    def __init__(self, pad_token=1, bidirectional=True, h_dim=300):
        super(Attention, self).__init__()
        self.bidirectional, self.h_dim, self.pad_token = bidirectional, h_dim, pad_token
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_e, out_e, out_d):
        '''Produces context with attention distribution'''

        # Deal with bidirectional encoder, move batches first
        if self.bidirectional: # sum hidden states for both directions
            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)
            
        # Move batches first
        out_e = out_e.transpose(0,1) # b x sl x hd
        out_d = out_d.transpose(0,1) # b x tl x hd

        # Dot product attention, softmax, and reshape
        attn = out_e.bmm(out_d.transpose(1,2)) # (b x sl x hd) (b x hd x tl) --> (b x sl x tl)
        attn = self.softmax(attn).transpose(1,2) # --> b x tl x sl

        # Get attention distribution
        context = attn.bmm(out_e) # --> b x tl x hd
        context = context.transpose(0,1) # --> tl x b x hd
        return context

#%%
# import torch
# import torch.nn  as nn
# embedding = torch.randn(max , 1, 300)
# init  =  torch.zeros(2, 1 , 256 )
# hidden  = ( init , init.clone())
# print(init.size())

# init size = ([ layers , batch_size , hidden_size])
# print( hidden.shape)

# embedding dim : ( vocab size , embedding dim : 300)
# memory size : [ (embedding_size(0) , batch_size , hidden_size: 256 )]
# h returned as a tuple of size( init.size() , init.size() )
# memory  , h = (nn.LSTM( 300 , 256 , 2 ))(embedding , hidden)
# print(memory.size())

# print(h[0].size())
# print(h[1])





