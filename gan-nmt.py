#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 00:51:55 2019

@author: shuvayan
"""




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
from collections import namedtuple

PROCESSED_PATH = '/Users/shuvayan/Desktop/Dataset/wmt15-de-en/'
train_de = open('/Users/shuvayan/Desktop/Dataset/wmt15-de-en/news-commentary-v10.de-en.de' , encoding ='utf-8').read().split('\n')
train_en = open('/Users/shuvayan/Desktop/Dataset/wmt15-de-en/news-commentary-v10.de-en.en' , encoding ='utf-8').read().split('\n')

batch_size = 100
embedding_size = 300
min_count = 2
device = torch.device('cuda')


Sentence = namedtuple('Sentence', ['index', 'tokens'])
# sentence = 'I am something great'
# a = Sentence( 0 ,sentence.split() )
# print(a)



def read_dataset(data):

    sentences = [Sentence(index , text.split()) for index , text in enumerate(data)]
    return sentences

train_de_sentences = read_dataset(train_de)
train_en_sentences = read_dataset(train_en)

UNK = '<UNK>'
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'


class VocabItem:

    def __init__(self, string, hash=None):
        """
        Our token object, representing a term in our vocabulary.
        """
        self.string = string
        self.count = 0
        self.hash = hash

    def __str__(self):
        """
        For pretty-printing of our object
        """
        return 'VocabItem({})'.format(self.string)

    def __repr__(self):
        """
        For pretty-printing of our object
        """
        return self.__str__()

class Vocab:

    def __init__(self, min_count=0, no_unk=False,
                 add_padding=False, add_bos=False,
                 add_eos=False, unk=None):
        self.no_unk = no_unk
        self.vocab_items = []
        self.vocab_hash = {}
        self.word_count = 0
        self.special_tokens = []
        self.min_count = min_count
        self.add_padding = add_padding
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.unk = unk

        self.UNK = None
        self.PAD = None
        self.BOS = None
        self.EOS = None

        self.index2token = []
        self.token2index = {}

        self.finished = False

    def add_tokens(self, tokens):
        if self.finished:
            raise RuntimeError('Vocabulary is finished')

        for token in tokens:
            if token not in self.vocab_hash:
                self.vocab_hash[token] = len(self.vocab_items)
                self.vocab_items.append(VocabItem(token))

            self.vocab_items[self.vocab_hash[token]].count += 1
            self.word_count += 1

    def finish(self):

        token2index = self.token2index
        index2token = self.index2token

        tmp = []

        if not self.no_unk:

            # we add/handle the special `UNK` token
            # and set it to have index 0 in our mapping
            if self.unk:
                self.UNK = VocabItem(self.unk, hash=0)
                self.UNK.count = self.vocab_items[self.vocab_hash[self.unk]].count
                index2token.append(self.UNK)
                self.special_tokens.append(self.UNK)

                for token in self.vocab_items:
                    if token.string != self.unk:
                        tmp.append(token)

            else:
                self.UNK = VocabItem(UNK, hash=0)
                index2token.append(self.UNK)
                self.special_tokens.append(self.UNK)

                for token in self.vocab_items:
                    if token.count <= self.min_count:
                        self.UNK.count += token.count
                    else:
                        tmp.append(token)
        else:
            for token in self.vocab_items:
                tmp.append(token)

        # we sort our vocab. items by frequency
        # so for the same corpus, the indices of our words
        # are always the same
        tmp.sort(key=lambda token: token.count, reverse=True)

        # we always add our additional special tokens
        # at the end of our mapping
        if self.add_bos:
            self.BOS = VocabItem(BOS)
            tmp.append(self.BOS)
            self.special_tokens.append(self.BOS)

        if self.add_eos:
            self.EOS = VocabItem(EOS)
            tmp.append(self.EOS)
            self.special_tokens.append(self.EOS)

        if self.add_padding:
            self.PAD = VocabItem(PAD)
            tmp.append(self.PAD)
            self.special_tokens.append(self.PAD)

        index2token += tmp

        # we update the vocab_hash for each
        # VocabItem object in our list
        # based on their frequency
        for i, token in enumerate(self.index2token):
            token2index[token.string] = i
            token.hash = i

        self.index2token = index2token
        self.token2index = token2index

        if not self.no_unk:
            print('Unknown vocab size:', self.UNK.count)

        print('Vocab size: %d' % len(self))

        self.finished = True


    def __getitem__(self, i):
        return self.index2token[i]

    def __len__(self):
        return len(self.index2token)

    def __iter__(self):
        return iter(self.index2token)

    def __contains__(self, key):
        return key in self.token2index

    def tokens2indices(self, tokens, add_bos=False, add_eos=False):
        """
        Returns a list of mapping indices by processing the given string
        with our `tokenizer` and `token_function`, and defaulting to our
        special `UNK` token whenever we found an unseen term.

        :param string: A sentence string we wish to map into our vocabulary.

        :param add_bos: If we should add the `BOS` at the beginning.

        :param add_eos: If we should add the `EOS` at the end.

        :return: A list of ints, with the indices of each token in the
                given string.
        """
        string_seq = []
        if add_bos:
            string_seq.append(self.BOS.hash)
        for token in tokens:
            if self.no_unk:
                string_seq.append(self.token2index[token])
            else:
                string_seq.append(self.token2index.get(token, self.UNK.hash))
        if add_eos:
            string_seq.append(self.EOS.hash)
        return string_seq

    def indices2tokens(self, indices, ignore_ids=()):
        """
        Returns a list of strings by mapping back every index to our
        vocabulary.

        :param indices: A list of ints.

        :param ignore_ids: An itereable with indices to ignore, meaning
                           that we will not look for them in our mapping.

        :return: A list of strings.

        Will raise a KeyException whenever we pass an index that we
        do not have in our mapping, except when provided with `ignore_ids`.

        """
        tokens = []
        for idx in indices:
            if idx in ignore_ids:
                continue
            tokens.append(self.index2token[idx].string)

        return tokens



src_vocab = Vocab(min_count=min_count, add_padding=True)

# for the output vocabulary
# we do not need the `UNK` token (we know all the classes), or the `PAD`
# tgt_vocab = Vocab(no_unk=True, add_padding=False)

tgt_vocab = Vocab(min_count=min_count, add_padding=True)



for sentence in train_en_sentences:
    src_vocab.add_tokens(sentence.tokens)
for sentence in train_de_sentences:
    tgt_vocab.add_tokens(sentence.tokens)

src_vocab.finish()
tgt_vocab.finish()


#%%
sentence1="SAN FRANCISCO – It has never been easy to have a rational conversation about the value of gold."
sentence2= "SAN FRANCISCO – Es war noch nie leicht, ein rationales Gespräch über den Wert von Gold zu führen."

sen1 = src_vocab.tokens2indices(sentence1.split())
sen2 = tgt_vocab.tokens2indices(sentence2.split())

class Embed(nn.Module):

    def __init__(self, vocab_size , embedding_dim ):
        super(Embed , self).__init__()
        self.layer = nn.Embedding( vocab_size , embedding_dim )

    def forward(self, x):
        output = self.layer(x)
        return output


SRC_VOCAB_SIZE = len(src_vocab)
TGT_VOCAB_SIZE =len(tgt_vocab)
embedding_dim =256

src_embedding_layer = Embed(SRC_VOCAB_SIZE , embedding_dim)
tgt_embedding_layer = Embed(TGT_VOCAB_SIZE , embedding_dim)

print(torch.LongTensor(sen1).shape)
print(len(sen2))

src_embed = src_embedding_layer(torch.LongTensor(sen1))
tgt_embed = tgt_embedding_layer(torch.LongTensor(sen2))

print(src_embed.shape)
print(tgt_embed.shape)

MAX_SIZE =max(src_embed.shape[0] , tgt_embed.shape[0])

def make_image( src, tgt):
    """
    

    Parameters
    ----------
    src : tensor of shape( max , emd dim)
        src_embeddings
    tgt : tensor of shape( max , embd_dim)
        tgt_embeddings

    Returns
    -------
    a 3d image

    """
    image = torch.zeros( max(src.shape[0] , tgt.shape[0]) , max(src.shape[0] , tgt.shape[0]) , src.shape[1] )

    for i  in range(src.shape[0]):
        slice_src = src[i, :]
        pro = slice_src*tgt
        image[i , : , :]=pro

    return image

image = make_image(src_embed, tgt_embed)
image  =image.permute(2, 0, 1)
image = image.unsqueeze(0)
print(image.shape)

def topk( k , data):
    """
    Parameters
    ----------
    k : int
        for getting top k values
    data : the data a tensor in sorted
        

    Returns
    -------
    top k values , their indices 

    """
    sort , idx =  data.sort(descending =True)
    return sort[:k] , idx[:k]


class NetD( nn.Module):

    def __init__(self, max_size):
        super( NetD , self).__init__()
        self.main = nn.Sequential(
                # input is embedding_dim * max * max
                nn.Conv2d( embedding_dim , embedding_dim*2 ,3, 1, 1 , bias =False),# 20 -2/2
                # input is embdding *2 , max , max
                nn.BatchNorm2d(embedding_dim*2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size =2),
                #embedding_dim*2 , max/2 , max/2
                nn.Conv2d(embedding_dim*2, embedding_dim*3 , 3 , 1, 1,bias =False),
                nn.BatchNorm2d(embedding_dim*3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                )
        self.fc1= nn.Linear( (int)(embedding_dim*3*(max_size//4) * (max_size//4)), max_size )
        self.fc2 =nn.Linear(max_size , TGT_VOCAB_SIZE)
        self.soft = nn.Softmax()

    def forward(self, x):
        x= self.main(x)
        # Flattening
        x = x.view(-1)
        # x --> embedding_dim*3 * max/4 * max/4
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x

model = NetD(MAX_SIZE)

output = model( image)
print(output)

_ , indxs = output.topk(MAX_SIZE)

string =""
for indx in indxs:
    string+=(tgt_vocab.index2token[indx].string) + ' '
print(string)
print("\n")
print(sentence2)
print('\n')




#%%









