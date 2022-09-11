import re
import nltk
import gensim
import numpy as np
import pandas as pd
nltk.download('punkt')
from typing import List, Tuple

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as st
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords = st.words('english')

########################## Text Filtering utilities

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py  emoticons list
# https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt Chat shortcuts

def remove_url_func(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r' ', text)

def remove_punctuations(text):
    punctuations = re.compile(r'[~`!@#$%^&*(,<،>){}\\/|\'"?؟_+-=~\[\]]')
    return punctuations.sub(r' ', text)

def remove_html_func(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r' ', text)

def remove_weird_chars(text):
    weridPatterns = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)
    patterns = [re.compile('\r'), re.compile('\n'), re.compile('&amp;')]
    text = weridPatterns.sub(r'', text)
    for p in patterns:
        text = p.sub(r' ', text)
    return text

def remove_extra_repeated_alpha(text):
    """
    Remove extra repeated alphabets in a word
    check these links:
    demo : https://regex101.com/r/ALxocA/1
    Question: https://bit.ly/2DoiPqS
    """
    return re.sub(r'([^\W\d_])\1{2,}', r'\1', text)


def preprocess_text(text, remove_urls=True, remove_html=True, 
             weird_patterns=True , remove_stopwords=True, 
             punctuations=True, remove_repeated_alpha=True):
    
    """
    Gets string text as input and preprocesses the text for 
    filtering urls, html, punctuations and stopwords and 
    returns it.
    
    """
    
    # lowercase the text.
    text = text.lower()
    
    # remove url
    if remove_urls:
        text = remove_url_func(text)
        
    # remove html tags
    if remove_html:
        text = remove_html_func(text)
        
    # remove emojis/symbols & pictographs/transport & map symbols/flags(iOS)
    if weird_patterns:
        text = remove_weird_chars(text)
    
    # remove punctuations
    if punctuations:
        text = remove_punctuations(text)
    
    # remove stop words
    # Removing stop words doesn't make issues for transformers.
    # https://datascience.stackexchange.com/a/87548/70008
    if remove_stopwords:
        temp = text.split(' ')
        temp = [i for i in temp if i not in stopwords]
        text = ' '.join(temp)
        
    # Alter words with repeated alphabets
    if remove_repeated_alpha:
        text = remove_extra_repeated_alpha(text)
    
    # remove extra spaces in the text.
    text = re.sub(' +', ' ', text)
    text = text.strip()
    
    # Return the remaining words in the sentence.
    # text = text.split(' ')
    return text


################################  Text-Related Utilities


def load_sents(caption_series: pd.Series):
    """
    Returns the tf-idf Vectorized captions
    """
    tfidf = TfidfVectorizer(use_idf=True, max_features=1000, tokenizer=word_tokenize)
    data = [preprocess_text(i) for i in caption_series]
    tfidf_train = tfidf.fit_transform(data)
    return tfidf_train

def load_df(filename: str):
    """
    Loads the videos path and preprocessed captions
    
    """
    df = pd.read_csv(filename)
    df.caption = df.caption.astype(str)
    sents = load_sents(df.caption)
    return df.path, sents

def train_gensim_model(final_words:List[List[str]]):
    """
    Trains a gensim model for vocabularies.
    
    """
    model = gensim.models.Word2Vec(window=20, min_count=1, workers=4)
    model.build_vocab(d, progress_per=100)
    model.train(final_words, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('gensim_model.w2v')
    return model


############################### DataLoader 2

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize2(x: List[str]):
    """
    Tokenize x
    Args:
        x(list): List of sentences/strings to be tokenized
    Returns:
        Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x)
    t=tokenizer.texts_to_sequences(x)
    return t, tokenizer

def pad2(x, length=None):
    """
    Pad x
    Args:
        x: List of sequences.
        length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    Returns:
        Padded numpy array of sequences
    """
    padding=pad_sequences(x,padding='post',maxlen=length)
    return padding

# def preprocess(sentences):
#     text_tokenized, text_tokenizer = tokenize(sentences)
#     text_pad = pad(text_tokenized)
#     return text_pad, text_tokenizer

def load_sents2(caption_series: pd.Series) -> list:
    """
    Loads vectorized sentences
    Args:
        caption_series(pd.Series): caption column of dataframe.
    Rerurns:
        
    
    """
    data = [preprocess_text(i) for i in caption_series]
    text_tokenized, text_tokenizer = tokenize2(data)
    test_pad = pad2(text_tokenized)
    return test_pad

def load_df2(filename: str)-> Tuple[pd.Series, list]:
    """
    Loads the videos path and preprocessed captions
    
    """
    df = pd.read_csv(filename)
    df.caption = df.caption.astype(str)
    vectorized_sents = load_sents2(df.caption)
    return df.path, vectorized_sents


