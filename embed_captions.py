"""
Embedding queries and captions (Jayashabari, Jamie)

process caption/queries by lowercasing text, stripping punctuation, tokenizing (refer to BagOfWords)
vocab
compute IDF for vocab where total caption count = N
function to embed caption text using GloVe word embeddings
document is a lit of captions
"""
from collections import Counter
import numpy as np
import re, string
from gensim.models.keyedvectors import KeyedVectors
import codecs
from nltk.tokenize import word_tokenize

punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))
 
 
def strip_punc(corpus):

    return punc_regex.sub('', corpus)

def process_text(text):
    """
    text can be both captions and queries
    lowercases, strips punctuation, tokenizes
    returns list of words in text
    """
    
    return (strip_punc(text)).lower().split()


def get_captions(class_coco):
    """
    forms vocabulary by
    getting all words across all captions in COCO dataset, basically creates vocabulary list
    """
     #return vocab
    #directly get captions from COCO class



def compute_idf(captions, counters):
    """
    takes in list of words in caption
    computes idf for each word in the caption list
    return list of idfs
    """
    #get vocab from coco class
    count = Counter()
    for caption in captions:
        caption = process_text(caption)
        caption = list(set(caption)) #prone to error!!!!!!!!!!!!!!!!! :)
        count.update(caption)
    count = dict(count)
    dict_of_idfs = defaultdict(None)
    for word in count.keys():
        N = len(captions)
        nt = count[word]
        dict_of_idfs[word]=np.log10(N / nt)
    return dict_of_idfs

""" create counter to maintan tally across all captions of given a word how many captions has that word appeared in, go caption by caption, lowercase everything, get unique words, use words and update counter, update counter for every caption
keys of counter represent whole vocabulary, now that counter is mapping of word to doc count, can easily convert that from count to frequency, 
create idf dictionary"""    
        
#want dictionary that maps word to idf
#can get vocabulary from going through captions
    
from gensim.models import KeyedVectors
filename = "glove.6B.200d.txt.w2v"

# this takes a while to load -- keep this in mind when designing your capstone project
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

def embed(caption):
    """
    embeds captions/queries
    compute_idf(caption_list) multiplied by glove embeddings added together
    word not in vocab should return embedding vector of 0s
    """
    word_embeddings = []
    for word in process_text(caption):
        if word not in vocab:
            return np.zeros(200,)
        else:
            word_embeddings.append(compute_idf(word)*glove[word])
    word_embeddings = np.array(word_embeddings.sum())
    
    return word_embeddings



    