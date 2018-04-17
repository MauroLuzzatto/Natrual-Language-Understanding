# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 08:08:48 2018

@author: Dario
"""

import numpy as np
from WordFrequency2_m5 import WordFrequencyDist_D


def preprocessing(pathToFile, training_mode, words_to_idx, word_dict):

    def build_vocabulary(raw_scentences):
        words = []
        for scent in raw_scentences:
            word_scent = scent.split()
            words.extend(word_scent)
        word_dist = WordFrequencyDist_D(words)
        # <unk>, <pad>, <eos>, <bos> are not words, therfore only 19996
        word_dict = {word_dist[i][0]: word_dist[i][1] for i in range(0, 19996)}
        return word_dict
        
    def Tokenize(raw_scentences, word_dict):
        """
        Takes raw scentences and the word dictinary as input and returns the 
        scentences with new formating:
            - max 30 words per scentence
            - <bos> and <eos> added at beginning and end of each scentence
            - word replaced with <unk> if not in dictionary
            - scentences paded with <pad> if shorter than 30 words
        """
        words_final = []
        tokenized = []
        batchsize = 64
        for scent in raw_scentences:
            word_scent = scent.split()
            for i in range(0,len(word_scent)):
                if not(word_scent[i] in word_dict):
                    word_scent[i] = '<unk>'
                    
            sentence = ['<bos>']
            if len(word_scent) <= 28:
                sentence.extend(word_scent)
                sentence.extend(['<eos>'])
                sentence.extend(['<pad>']*(28-len(word_scent)))
                
                tokenized.append(sentence)
                words_final.extend(sentence)
        
        nr_of_dummy_scentences = batchsize -  len(tokenized)%batchsize
        print('len tok ', len(tokenized))
        print('nr dummy scent ', nr_of_dummy_scentences)
        for _ in range(0,nr_of_dummy_scentences):
            tokenized.append(['<pad>']*30)
                
        return tokenized, words_final

    # load data
    text_file_object = open(pathToFile,'r')
    raw_scentences = text_file_object.readlines()
    text_file_object.close()
             

    if training_mode:
        print('preprocessing for training data')
        # build up the dictionary
        word_dict = build_vocabulary(raw_scentences)   
        
        new_scentences, words_final = Tokenize(raw_scentences, word_dict)
        
        word_dist_final = WordFrequencyDist_D(words_final)
        words_to_idx = {word_dist_final[indx][0]: indx for indx in range(0, 20000)}
        ## Create the training data
        X = [[words_to_idx[w] for w in sent[:-1]] for sent in new_scentences]
        y = [[words_to_idx[w] for w in sent[1:]] for sent in new_scentences]
        
    
    else:
        print('preprocessing for test data')
        # process the test data
        new_scentences, words_final = Tokenize(raw_scentences, word_dict)
        
        X = [[words_to_idx[w] for w in sent[:-1]] for sent in new_scentences]
        y = [[words_to_idx[w] for w in sent[1:]] for sent in new_scentences]
    
    return X,y, words_to_idx, word_dict


    