#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:41:33 2019

@author: shuvayan
"""

'''
Based on 
https://bastings.github.io/annotated_encoder_decoder/

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tensorboardX import SummaryWriter

logdir = '/media/nibaran/STORAGE/Shuvayan_Sandip_NLPs/translation'

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') if USE_CUDA else torch.device('cpu') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


'''
Encoder reads in the source sentence and produces  a sequence of hidden states h1, h2, .. hm , one for each word.
This should capture the maeaning of the word in context of its sentence
A bi -RNN is used as the encoder.
First of all , all the source words are embed , we look up the word embedding in a lookup table.We denote the 
word embedding for word i  with xi.
A forward GRU reads the sentence from left tom right , hj = GRU( xj , hj-1) , we obtain the new hidden state from the old hidden state
and current word embedding.
By concatenating the hidden state of both the GRU , we get the hidden state of the word with context of the entire sentence.

Decoder :
    si = f( si-1 ,yi-1 , c )
    An attention mechanism dynamically selects that part of the source sentence which is most relevant for pedicting 
    the current target word. Comparing last decoder state ,and each source hidden state. 

'''


class EncoderDecoder( nn.Module):
    '''
        A standard encoder -decoder architecture
    '''
    def __init__(self , encoder , decoder , src_embed , tar_embed, generator):
        super(EncoderDecoder , self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tar_embed = tar_embed
        self.generator =generator

    def forward(self , src , tar , src_mask , trg_mask , src_lens , tar_lens):
        encoder_hidden  , encoder_final = self.encode( src , src_mask , src_lens)
        return self.decode( encoder_hidden ,encoder_final , src_mask , tar, trg_mask  )

    def encode ( self, src , src_mask , src_len):
        return ( self.encoder( self.src_embed(src) , src_mask ,src_len))

    def decode ( self , encoder_hidden , encoder_final , src_mask , tar, tar_mask , decoder_hidden  =None):
        return ( self.decoder (self.tar_embed(tar) , encoder_hidden ,encoder_final , src_mask , tar_mask , hidden =decoder_hidden  ) )


class Generator(nn.Module):
    '''
    Defining standard linear + softmax generation step
    
    '''
    def __init__(self ,hidden_size ,vocab_size ):
        super(Generator , self).__init__()
        self.proj = nn.Linear( hidden_size , vocab_size , bias= False)

    def forward(self , x):
        return F.log_softmax(self.proj(x) , dim=-1)

class Encoder(nn.Module):
    '''
    Encodes a sequence of word embeddings
    '''
    def __init__(self, input_size , hidden_size , num_layers =1 ,dropout =0. ):
        super(Encoder , self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size , hidden_size , num_layers , batch_first =True , bidirectional =True , dropout = dropout)

    def forward( self, x , mask , lengths):
        '''
        Parameters
        ----------
        x : tensor
            (sequence of word embeddings) dim: [ batch , time, dim]
        mask : 
            DESCRIPTION.
        lengths : TYPE
            DESCRIPTION.

        Returns
        -------
        Output tensor and final tensor
        Final tensor = Concatenated tensor from hidden states of both dir of GRU

        '''

        '''
        Why packing is required ? 
        
        For the unpacked case, the directions can be separated using 
        output.view(seq_len, batch, num_directions, hidden_size), 
        with forward and backward being direction 0 and 1 respectively.
        
        For 
        '''
        # packs sequence of data and due to batch first = True , changes dim to [ batch , time , dim]
        packed = pack_padded_sequence(x , lengths , batch_first = True)
        # returns output and hidden state from Gru
        # output ( dim ) : ( time , batch , num_dir * hidden_size)
        # final ( num_layers* num_dir , batch , hidden_size )
        output , final = self.rnn(packed)
        output , _ = pad_packed_sequence( output , batch_first =True)

        # Manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final , bwd_final] , dim =2) #[num_layers , batch , 2*dim]
        return output , final




class Decoder( nn.Module):
    '''
        A conditional rnn decoder with attention
        TEACHER FORCING
        In train loop , a for loop computes the hidden decoder states one step at a  time . During training we know exactly what the target 
        words should be( tar_embd,) .We simply feed the correct previous target word embedding to the GRU at each time step.This is called teacher forcing
        
        The forward function returns all decoder hidden states and pre-output vectors
        
        For prediction time the forward funcion is used for a single time step , after predicting a word from the pre - output vector we can call it again , 
        supplying the word embedding of the previously predicted word and last state.
        
    '''

    def __init__(self ,embd_size , hidden_size , attention , num_layers =1 , dropout =0.5 , bridge =True   ):
        super( Decoder , self ).__init__()
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.attention = attention
        self.num_layers =num_layers
        self.dropout = dropout
        self.bridge = nn.Linear( 2*hidden_size , hidden_size , bias =True) if bridge == True else None
        self.dropout_layer = nn.Dropout( p =dropout)
        self.rnn = nn.GRU(embd_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.pre_output_layer = nn.Linear( hidden_size + 2 *hidden_size + embd_size ,hidden_size , bias = False )


    def forward_step(self , prev_embd , encoder_hidden , src_mask , proj_key , hidden):
        '''
        
        Performs a single decoder step ( 1 word )
        Parameters
        ----------
        prev_embd : TYPE
            DESCRIPTION.
        encoder_hidden : TYPE
            DESCRIPTION.
        src_mask : TYPE
            DESCRIPTION.
        proj_key : TYPE
            DESCRIPTION.
        hidden : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        query = hidden[-1].unsqueeze(1) # ( layers , B , D) -> ( B , 1 , D)
        context , atten_probs = self.attention( query = query , proj_key = proj_key , value= encoder_hidden , mask = src_mask)

        # updating rnn state

        rnn_input = torch.cat( [prev_embd , context ] , dim =2)
        output , hidden = self.rnn( rnn_input , hidden)

        pre_output = torch.cat( [ prev_embd , output, context ] , dim =2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden , pre_output

    def forward( self , trg_embed , encoder_hidden , encoder_final , src_mask , trg_mask ,hidden = None ,  max_len =None ):
        '''
        Unroll the decoder one step at  a time

        Parameters
        ----------
        trg_embed : TYPE
            DESCRIPTION.
        encoder_hidden : TYPE
            DESCRIPTION.
        encoder_final : TYPE
            DESCRIPTION.
        src_mask : TYPE
            DESCRIPTION.
        trg_mask : TYPE
            DESCRIPTION.
        hidden : TYPE, optional
            DESCRIPTION. The default is None.
        max_len : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''

        if max_len is None:
            max_len = trg_mask.size(-1)

        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)


        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas






def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))

    return model.cuda() if USE_CUDA else model


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()


def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0
    writer = SummaryWriter(log_dir =logdir )
    for i, batch in enumerate(data_iter, 1):
        
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            writer.add_scalar('loss' , (loss/ batch.nseqs))
            start = time.time()
            print_tokens = 0
    writer.close()

    return math.exp(total_loss / float(total_tokens))




def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
          np.random.randint(1, num_words, size=(batch_size, length)))
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)



class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)
  

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]



def print_examples(example_iter, model, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"    
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break


def train_copy_task():
    """Train the simple copy task."""
    num_words = 11
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100))
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad(): 
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples(eval_data, model, n=2, max_len=9)
        
    return dev_perplexities

dev_perplexities = train_copy_task()

def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    
plot_perplexity(dev_perplexities)

#%%


#from torchtext import data, datasets
#import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import math, copy, time
#import matplotlib.pyplot as plt
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#
#if True:
#    import spacy
#    spacy_de = spacy.load('de')
#    spacy_en = spacy.load('en')
#
#    def tokenize_de(text):
#        return [tok.text for tok in spacy_de.tokenizer(text)]
#
#    def tokenize_en(text):
#        return [tok.text for tok in spacy_en.tokenizer(text)]
#
#    UNK_TOKEN = "<unk>"
#    PAD_TOKEN = "<pad>"    
#    SOS_TOKEN = "<s>"
#    EOS_TOKEN = "</s>"
#    LOWER = True
#
#
#    # we include lengths to provide to the RNNs
#    SRC = data.Field(tokenize=tokenize_de, 
#                     batch_first=True, lower=LOWER, include_lengths=True,
#                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)
#    TRG = data.Field(tokenize=tokenize_en, 
#                     batch_first=True, lower=LOWER, include_lengths=True,
#                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
#
#    MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
#    train_data, valid_data, test_data = datasets.IWSLT.splits(
#        exts=('.de', '.en'), fields=(SRC, TRG), 
#        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
#            len(vars(x)['trg']) <= MAX_LEN)
#    MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
#    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
#    TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)
#    
#    PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]
#
#
#def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
#    """ This prints some useful stuff about our data sets. """
#
#    print("Data set sizes (number of sentence pairs):")
#    print('train', len(train_data))
#    print('valid', len(valid_data))
#    print('test', len(test_data), "\n")
#
#    print("First training example:")
#    print("src:", " ".join(vars(train_data[0])['src']))
#    print("trg:", " ".join(vars(train_data[0])['trg']), "\n")
#
#    print("Most common words (src):")
#    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
#    print("Most common words (trg):")
#    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")
#
#    print("First 10 words (src):")
#    print("\n".join(
#        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
#    print("First 10 words (trg):")
#    print("\n".join(
#        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")
#
#    print("Number of German words (types):", len(src_field.vocab))
#    print("Number of English words (types):", len(trg_field.vocab), "\n")
#    
#    
##print_data_info(train_data, valid_data, test_data, SRC, TRG)
#'''
#Data set sizes (number of sentence pairs):
#train 143121
#valid 690
#test 963 
#
#First training example:
#src: david gallo : das ist bill lange . ich bin dave gallo .
#trg: david gallo : this is bill lange . i 'm dave gallo . 
#
#Most common words (src):
#         .     138289
#         ,     105953
#       und      41845
#       die      40813
#       das      33324
#       sie      33035
#       ich      31150
#       ist      31036
#        es      27449
#       wir      25820 
#
#Most common words (trg):
#         .     137219
#         ,      91622
#       the      73348
#       and      50279
#        to      42801
#         a      39577
#        of      39498
#         i      33522
#        it      32920
#      that      32645 
#
#First 10 words (src):
#00 <unk>
#01 <pad>
#02 </s>
#03 .
#04 ,
#05 und
#06 die
#07 das
#08 sie
#09 ich 
#
#First 10 words (trg):
#00 <unk>
#01 <pad>
#02 <s>
#03 </s>
#04 .
#05 ,
#06 the
#07 and
#08 to
#09 a 
#
#Number of German words (types): 15768
#Number of English words (types): 13004 
#'''
#
#DEVICE ='cpu'
#train_iter = data.BucketIterator(train_data, batch_size=64, train=True,
#                                 sort_within_batch=True, 
#                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
#                                 device=DEVICE)
#valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, 
#                           device=DEVICE)
## print(type(train_iter))
## print(type(valid_iter))
## print(type(SRC))
#%%
def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch((batch.src ), (batch.trg), pad_idx)

def train(model, num_epochs=10, lr=0.0003, print_every=100 , train_iter =None, valid_iter=None , SRC=None , TRG=None, PAD_INDEX= None):
    """Train a model on IWSLT"""
    
    if USE_CUDA:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    dev_perplexities = []

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
                                     model,
                                     SimpleLossCompute(model.generator, criterion, optim),
                                     print_every=print_every)
        
        model.eval()
        with torch.no_grad():
            print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), 
                           model, n=3, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)        

            dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), 
                                       model, 
                                       SimpleLossCompute(model.generator, criterion, None))
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)
        
    return dev_perplexities

#model = make_model(len(SRC.vocab), len(TRG.vocab),
#                   emb_size=256, hidden_size=256,
#                   num_layers=1, dropout=0.2)
#dev_perplexities = train(model, print_every=100)



