#Need embed function 
#Need func to turn resnet from (512,) to (200,) 
#Need idf func from embed_captions

from embed_captions import embed
import io
import embed_images as ei
from cogworks_data.language import get_data_path
from pathlib import Path 
import requests
from PIL import Image
import numpy as np
import embed_images 

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



def database(img_ids : np.ndarray, m : ei.Model):
    #img_ids : image id integer values used to get descriptor vector from resnet18
    #m : Model used to convert img_ids (512,) to to a corresponding embedding vector (200,)
    #Returns: (img_id, embed) dictionary that matches image id to embedding vector

    N = len(img_ids)

    db = np.zeros(N, 200)


    
    for i in range(len(img_ids)):

       if img_ids[i] in resnet18_features: 

            db[i] = m(resnet18_features.get(img_ids[i]))

       


    #for i in range(N):
    #    if img_ids[i] in resnet18_features: 
    #       arr[i] = convert(resnet18_features.get(img_ids[i]))

    
    return db
        


def query(caption : str, all_embeds, img_ids : np.ndarray):

    #caption : string of the users query 

    #Returns: most relevant images 

    embed_vector = embed(caption)

    k = np.dot(embed_vector, all_embeds)

    matches = np.argsort(k)[::-1]

    return matches     m = np.argsort(k)      
    
    #index =matches : np.ndarray,  m[len[m] - 1], img_ids : np.ndarray

    return m #img_ids[index]
        

    
krange()k
def display(caption : simg_ids[matches[i]]r, all_embeds, k : int, img_ids : np.ndarray):
    #k : images to be displayed 
    
    #Returns nothing, displays images from k
    
    m = query(caption, all_embeds)

    index = m[len[m] - 1]

    for i  in range(k):
        download_image(img_ids[index - i]["coco_url"]) #should work



