#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:46:58 2019

@author: shuvayan sandip
"""



#%%
from torchtext import data, datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from torchtext.data import Field
from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') if USE_CUDA else torch.device('cpu') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)
CUDA_LAUNCH_BLOCKING =1
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
path ='/media/nibaran/STORAGE/Shuvayan_Sandip_NLPs/translation/parallel/'

exts =[ 'IITB.en-hi.en' , 'IITB.en-hi.hi']

tokenize = lambda x: x.split()
# To add stopwords stopwords =[]
MIN_FREQ =5
SRC = Field(sequential=True, tokenize=tokenize, lower=True, init_token='<s>' ,eos_token='</s>'   )
TRG = Field(sequential=True, tokenize=tokenize, lower=True, init_token='<s>' ,eos_token='</s>'   )
fields =[SRC , TRG]


class getDataset(datasets.TranslationDataset):
    name ='translation'
    urls=[]
    dirname='/media/nibaran/STORAGE/Shuvayan_Sandip_NLPs/translation/parallel/'
    def __init__(self , path , exts , fields):
        super(getDataset , self).__init__(path , exts , fields)


#things to pass
#examples, fields, filter_pred=None
'''
examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
'''

#https://github.com/pytorch/text/blob/284a51651dd9697f9afd76f2ceb23a8181ae7552/torchtext/datasets/translation.py#L10
dataset = getDataset(path , exts , fields)
train_data , val  , test = dataset.split(split_ratio =[0.8 , 0.1 ,0.1 ]   )
SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)

def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
    """ This prints some useful stuff about our data sets. """

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))
    print('test', len(test_data), "\n")

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['src']))
    print("trg:", " ".join(vars(train_data[0])['trg']), "\n")

    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
    print("Most common words (trg):")
    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
    print("First 10 words (trg):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of Hindi words (types):", len(src_field.vocab))
    print("Number of English words (types):", len(trg_field.vocab), "\n")
    
    
#print_data_info(train_data, val, test, SRC, TRG)
'''
Data set sizes (number of sentence pairs):
train 1244590
valid 155574
test 155574 

First training example:
src: 17th
trg: १७वाँ 

Most common words (src):
       the     937980
        of     565960
       and     482584
        to     384012
        in     303314
         a     271406
        is     222084
       for     149264
         -     144505
      that     137168 

Most common words (trg):
        के     601867
        और     427083
       में     394380
        की     341175
        से     272289
        है     250021
        को     247733
        का     220872
        कि     149619
       है।     149610 

First 10 words (src):
00 <unk>
01 <pad>
02 <s>
03 </s>
04 the
05 of
06 and
07 to
08 in
09 a 

First 10 words (trg):
00 <unk>
01 <pad>
02 <s>
03 </s>
04 के
05 और
06 में
07 की
08 से
09 है 

Number of English words (types): 92444
Number of Hindi words (types): 96256 

'''


#%%
import rnn
from rnn import EncoderDecoder , Encoder , Decoder , Generator , BahdanauAttention , make_model , Batch ,data_gen , SimpleLossCompute , greedy_decode
from rnn import lookup_words , print_examples , plot_perplexity , rebatch , train

#dev_perplexities = train_copy_task()
#plot_perplexity(dev_perplexities)


train_iter = data.BucketIterator(train_data, batch_size=256, train=True,
                                 sort_within_batch=True, 
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=DEVICE)
valid_iter = data.Iterator(val, batch_size=1, train=False, sort=False, repeat=False, 
                           device=DEVICE)

    
PAD_INDEX = TRG.vocab.stoi["<pad>"]
model = make_model(len(SRC.vocab), len(TRG.vocab),
                   emb_size=256, hidden_size=256,
                   num_layers=1, dropout=0.2)
dev_perplexities = train(model, print_every=100 , train_iter =train_iter , valid_iter=valid_iter ,SRC= SRC , TRG=TRG , PAD_INDEX =PAD_INDEX)

import sacrebleu

references = [" ".join(example.trg) for example in val]
print(references[0])

hypothesis=[]
alphas =[]

for batch in valid_iter:
	batch = rebatch(PAD_INDEX , batch)
	pred , attention = greedy_decode(model ,batch.src , batch.src_mask , batch.src_lengths , max_len =25 ,sos_index = TRG.vocab.stoi['<s>'] ,eos_index = TRG.vocab.stoi['</s>'])
	hypothesis.append(pred)
	alphas.append(attention)





hypothesis = [lookup_words(x , TRG.vocab) for x in hypothesis]
hypothesis =[" ".join(x) for x in hypothesis]
bleu=sacrebleu.raw_corpus_bleu(hypothesis, [references], .01).score
print(bleu)

  




