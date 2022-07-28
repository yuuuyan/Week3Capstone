#Need embed function 
#Need func to turn resnet from (512,) to (200,) 
#Need idf func from embed_captions

from embed_captions import embed
import io

from embed_images import w_embed
from cogworks_data.language import get_data_path
from pathlib import Path
import requests
from PIL import Image
import numpy as np

import pickle
with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)


def download_image(img_url: str) -> Image:
    """Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))



#takes in N image_ids to image embeddings (from resnet)

def convert(img_desc):
    #img_desc : image descriptor vector with a shape of (512,)
    #w_embed : temporary name for embed function

    matrix = img_desc * w_embed(img_desc)

    return matrix 



def database(img_ids : np.ndarray):
    #img_ids : image id integer values used to get descriptor vector from resnet18

    #Returns: (N, 512) array that matches image id to descriptor vector

    N = len(img_ids)

    arr = np.zeros(N, 512)

    
    for i in range(N):
        if img_ids[i] in resnet18_features: 
            arr[i] = resnet18_features.get(img_ids[i])

    
    return arr
        

def query(caption : str, all_embeds):

    #caption : string of the users query 

    #Returns: k most relevant images 

    embed_vector = embed(caption)

    k = np.dot(embed_vector, all_embeds)


def display(k : np.ndarray):
    #k : images to be displayed 
    
    #Returns nothing, displays images from k

    for i in k:
        download_image(i["coco_url"]) #should work



