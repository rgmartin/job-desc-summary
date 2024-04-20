"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from utils import read_corpus, pad_sents


class VocabEntry(object):
    def __init__(self, word2id=None) -> None:
        """
        @param word2id (dict): dictionary mapping words to indices
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0 # Pad token
            self.word2id['<s>'] = 1 # Start token
            self.word2id['</s>'] = 2 # End token
            self.word2id['<unk>'] = 3 # Unknown token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    def __getitem__(self,word):
        # Returns the index (int) of word(str) if it is in the vocabulary, else return unk_id
        return self.word2id.get(word, self.unk_id)
    
    def __contains__(self,word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)
    
    def __repr__(self):
        return f'Vocabulary[size = {len(self)}]'
    
    def id2word(self, wid):
        return self.id2word[wid]
    
    def add(self, word):
        if word not in self:
            wid = self.word2id[word ]  = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]
    
    def words2indices(self, sents):
        """
        # Convert list of words or list of sentences of words
        # into list or list of list of indices
        @param sents(list[str] or list[list[str]]: sentence(s) in words
        @return word_ids (list[int] or list[list[str]]): sentence(s) in indices
        """
        if isinstance( type(sents[0]) , List):
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]
        
    def indices2words(self, word_ids):
        """
        @param word_ids (list[int]): list of word_ids
        @return sents(list[str])
        """
        return [self.id2word[i] for i in word_ids]

    def to_input_tensor(self, sents:List[List[str]], device:torch.device) -> torch.Tensor:
        """Convert list of sentences (words) into tensor with necessary padding for shorter senteces

        @param sents (List[List[str]]): list of sentences(words)
        @param device

        @return sents_var: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry