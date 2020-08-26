import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchtext.data import Field
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import logging
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import DataLoader, Dataset




def get_iterator(dataset, batch_size, train=True,
                 shuffle=True, repeat=False):
    
    device = torch.device('cuda:0' if torch.cuda.is_available()
                          else 'cpu')
    
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    
    return dataset_iter




###
SEED = 42
###

LOGGER = logging.getLogger('tweets_dataset')

def train_val_split():
    """
    Stratified train test split by keyword.
    See https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert/comments#1.-Keyword-and-Location
    for explanation of choosing such validation strategy
    """

    X = pd.read_csv("data/prepared_df_train.csv")
    y = X['target'].values
    keyword = X['keyword'].values
    X = X.drop('target', axis=1)
    columns = X.columns
    X = X.values
    
    global SEED
    X_train, X_val, y_train, y_val = \
    train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=keyword)

    
    X_train = pd.DataFrame(X_train, columns=columns)
    y_train = pd.DataFrame(y_train, columns=['target'])

    train = pd.concat([X_train, y_train], axis=1)
    train.to_csv("data/train_train.csv", index=False)


    X_val = pd.DataFrame(X_val, columns=columns)
    y_val = pd.DataFrame(y_val, columns=['target'])

    val = pd.concat([X_val, y_val], axis=1)
    val.to_csv("data/val_val.csv", index=False)


def get_dataset(fix_length=100, lower=False, vectors=None):
    
    if vectors is not None:
        lower=True
        
    LOGGER.debug('Preparing CSV files...')
    # prepare_csv(train, test)


    
    TEXT = data.Field(sequential=True, 
                      lower=True, 
                      include_lengths=True, 
                      batch_first=True, 
                      fix_length=25)
    NUM_FEATURE = data.Field(use_vocab=False,
                       sequential=False,
                       dtype=torch.float16)
    KEYWORD = data.Field(use_vocab=True,
                         sequential=True)

    LOCATION = data.Field(use_vocab=True,
                          sequential=True)


    ID = data.Field(use_vocab=False,
                    sequential=False,
                    dtype=torch.float16)

    # LABEL = data.LabelField(dtype = torch.float)
    LABEL = data.Field(use_vocab=True,
                       sequential=False,
                       dtype=torch.float16)
    
    tv_datafields = [
        ("id", None), # we won't be needing the id, so we pass in None as the field
        ("keyword", None),
        ("location", None),
        ("text", TEXT),
        ("word_count", NUM_FEATURE),
        ("char_count", NUM_FEATURE),
        ("stop_word_count", NUM_FEATURE),
        ("punctuation_count", NUM_FEATURE),
        ("mention_count", NUM_FEATURE),
        ("hashtag_count", NUM_FEATURE),
        ("target", LABEL)]
        


    
    LOGGER.debug('Reading train csv files...')

    train_temp, val_temp = data.TabularDataset.splits(
        path='data/', format='csv', skip_header=True,
        train='train_train.csv', validation='val_val.csv',
        fields=tv_datafields
    )
    
    LOGGER.debug('Reading test csv file...')


    test_temp = data.TabularDataset(
        path='data/prepared_df_test.csv', format='csv',
        skip_header=True,
        fields=tv_datafields[:-1]
    )
    
    LOGGER.debug('Building vocabulary...')

    MAX_VOCAB_SIZE = 25000

    # TODO: проверить, нет ли здесь лика,
    # когда строю словарь по валидационной и тестовой выборках?
    TEXT.build_vocab(
        train_temp, val_temp, test_temp,
        max_size=MAX_VOCAB_SIZE,
        min_freq=10,
        vectors=GloVe(name='6B', dim=300)  # We use it for getting vocabulary of words
    )


    LABEL.build_vocab(train_temp)


    # KEYWORD.build_vocab(
    #     train_temp, val_temp, test_temp,
    #     max_size=MAX_VOCAB_SIZE,
    # )


    # LOCATION.build_vocab(
    #     train_temp, val_temp, test_temp,
    #     max_size=MAX_VOCAB_SIZE,
    # )


    
    word_embeddings = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)
    
    train_iter = get_iterator(train_temp, batch_size=32, 
                              train=True, shuffle=True,
                              repeat=False)
    val_iter = get_iterator(val_temp, batch_size=32, 
                            train=True, shuffle=True,
                            repeat=False)
    test_iter = get_iterator(test_temp, batch_size=32, 
                             train=False, shuffle=False,
                             repeat=False)
    
    
    LOGGER.debug('Done preparing the datasets')
    
    return TEXT, vocab_size, word_embeddings, train_iter, val_iter, test_iter
