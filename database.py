#Need embed function 
#Need func to turn resnet from (512,) to (200,) 
#Need idf func from embed_captions

import io


import requests
from PIL import Image


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
s

    #img_ids : image id integer value used to get descriptor vector from resnet18

    #Returns: (N, 512) array that matches image id to descriptor vector

    N = len(img_ids)

    arr = np.zeros(N, 512)

    
    for i in range(N):
        if img_ids[i] in resnet18_features: 
            arr[i] = resnet18_features.get(img_ids[i])

    
    return arr
        

 tion

    #caption : string of the users query 

    #Returns: k most relevant images 

    IDF(caption)

def display(k : np.ndarra):
    #k : images to be displayed 

def query(cap_emb):
