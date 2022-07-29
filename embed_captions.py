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
from cogworks_data.language import get_data_path
from pathlib import Path
from organizeCOCO import COCO

punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))
coco_temp = COCO()
 
def strip_punc(corpus):
    return punc_regex.sub('', corpus)

def process_text(text):
    """
    text: type String, text can be both captions and queries
    returns: list of words in text

    lowercases, strips punctuation, tokenizes
    """
    return (strip_punc(text)).lower().split(" ")

def get_captions(image_id):
    """
    image_id: id of image we want
    gets all captions of image
    returns list of captions from COCO class
    """
    return coco_temp.I_To_C(image_id)

def compute_idf(captions):
    """
    captions: list of strings representing all captions of an image
    goes through each caption in list of captions
    computes idf for each word in the caption
    returns word: idf dictionary
    """
    #get vocab from coco class
    count = Counter(coco_temp.get_Vocab()) # Getting vocab from COCO class instance "coco_temp"
    dict_of_idfs = {}

    for caption in captions:
        caption = process_text(caption)
        count.update(set(caption))
    
    for word in count.keys():
        N = len(captions)
        nt = count[word]
        dict_of_idfs[word]=np.log10(N / nt)

    return dict_of_idfs

def compute_idf_2(captions: dict):
    """
    captions: list of strings representing all captions of an image
    goes through each caption in list of captions
    computes idf for each word in the caption
    returns word: idf dictionary
    """
    #get vocab from coco class
    count = Counter() # Getting vocab from COCO class instance "coco_temp"
    dict_of_idfs = {}

    for caption in captions.values():
        caption = process_text(caption)
        count.update(caption)
    
    for word in count.keys():
        N = len(captions)
        nt = count[word]
        dict_of_idfs[word]=np.log10(N / nt)

    return dict_of_idfs


    
filename = "glove.6B.200d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

def embed(text, idfs):
    """
    text: String representing individual caption/query
    goes through each word in caption
    if word not in vocab should add vector of 0s
    else, multiplies each word's idf by its glove embedding
    adds everything
    returns caption embedding
    """
    word_embeddings = []
    for word in process_text(text):
        if word not in idfs or word not in glove:
            word_embeddings.append(np.zeros(200,))
        else:
            word_embeddings.append(idfs[word]*glove[word])
    word_embeddings = np.array(sum(word_embeddings))
    
    return word_embeddings



    