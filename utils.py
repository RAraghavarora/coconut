# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np

from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_nlg(predictions, references):
    # Initialize metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    # Compute scores
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    bleu_scores = 0
    
    for pred, ref in zip(predictions, references):
        # Compute ROUGE
        rouge_result = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += rouge_result[key].fmeasure
        
        # Compute BLEU
        ref_tokens = nltk.word_tokenize(ref.lower())
        pred_tokens = nltk.word_tokenize(pred.lower())
        bleu_scores += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
    
    # Average scores
    total = len(predictions)
    for key in rouge_scores:
        rouge_scores[key] /= total
    bleu_scores /= total
    
    return {'rouge': rouge_scores, 'bleu': bleu_scores}