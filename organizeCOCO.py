from cogworks_data.language import get_data_path
from pathlib import Path
from collections import defaultdict
import json
import pickle
import re, string
import random
import numpy as np

from numpy import empty

def strip_punc(corpus):
    punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))
    return punc_regex.sub('', corpus)

class COCO:

    def __init__(self) -> None:

        '''

        Loading the MSCOCO 2014 data set and creating dictionaries for 
        ease of access

        '''

        # load Resnet Data

        with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
            resnet18_features = pickle.load(f)

        # load COCO metadata

        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            self.data = json.load(f)

        # Creating the Image --> List of Captions dictionary

        self.image_to_caps = defaultdict(list)

        for caption in self.data["annotations"]:
            if caption["image_id"] in resnet18_features:
                self.image_to_caps[caption["image_id"]].append(caption["id"])
        
        # Creating the Image --> URLs dictionary

        self.image_to_URLs = defaultdict(None)

        for image in self.data["images"]:
            if image["id"] in resnet18_features:
                self.image_to_URLs[image["id"]] = image["coco_url"]

        # Creating the Caption --> Image dictionary

        self.cap_to_image = defaultdict(None)

        for caption in self.data["annotations"]:
            if caption["image_id"] in resnet18_features:
                self.cap_to_image[caption["id"]] = caption["image_id"]

        # Creating the Caption --> Actual Caption dictionary

        self.cap_to_cap = defaultdict(None)

        for caption in self.data["annotations"]:
            if caption["image_id"] in resnet18_features:
                self.cap_to_cap[caption["id"]] = caption["caption"]

        # Creating List of Unique Vocabulary
        
        self.vocab = []

        for caption in self.data["annotations"]:
            if caption["image_id"] in resnet18_features:
                text = (strip_punc(caption["caption"])).lower().split()
                for word in text:
                    if word not in self.vocab:
                        self.vocab.append(word)
                    else:
                        pass

    
    def getURL(self, image_id):
        """
        Returns the URL of an image given an Image ID

        Parameters:
        --------------------------------------------
        image_id: int
            The integer id of the image whose captions you wish to return
        
        Returns:
        --------------------------------------------
        URL: string
            URL of corresponding image
        """

    def I_To_C(self, image_id : int):

        """
        Returns a List of Captions given an Image ID

        Parameters:
        --------------------------------------------
        image_id: int
            The integer id of the image whose captions you wish to return
        
        Returns:
        --------------------------------------------
        List[int]
            List of corresponding caption ids as integers
        """

        return self.image_to_caps[image_id]

    def C_To_I(self, cap_id : int):

        """
        Returns an Image ID given an Caption ID

        Parameters:
        --------------------------------------------
        cap_id: int
            The integer id of the caption whose corresponding Image ID you wish to return
        
        Returns:
        --------------------------------------------
        ID: int
            Corresponding Image ID

        """
        return self.cap_to_image[cap_id]

    def rand_cap(self, image_id):
        """
        Returns a random caption for a given image

        Returns:
        --------------------------------------------
        Caption: [str]
            Random Caption for a given image  

        """

        caps = self.image_to_caps[image_id]

        if not caps:
            return ""
        
        return self.C_To_C(caps[np.random.randint(0, len(caps))])


    def C_To_C(self, cap_id : int):

        """
        Returns the actual string caption given an Caption ID

        Parameters:
        --------------------------------------------
        cap_id: int
            The integer id of the caption whose corresponding Image ID you wish to return
        
        Returns:
        --------------------------------------------
        Caption: str
            Corresponding Caption 

        """

        return self.cap_to_cap[cap_id]

    def get_Vocab(self):
        """
        Returns a list of all vocab words used among all captions
        in the data set
    
        Returns:
        --------------------------------------------
        Vocab: List[str]
            List of vocab words  

        """

        return self.vocab
    
    def return_ids(self):
        """
        Returns a list of all Image ids as a numpy array

        Returns:
        --------------------------------------------
        IDS: numpy.array[ints]
            List of vocab words  

        """

        ID_list = list(self.image_to_caps.keys())
        return np.array(ID_list)
    
    # def get_all_captions(self, ):



        



        
        

        
