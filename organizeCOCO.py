from cogworks_data.language import get_data_path
from pathlib import Path
from collections import defaultdict
import json
import re, string

def strip_punc(corpus):
    return punc_regex.sub('', corpus)

class COCO:

    def __init__(self) -> None:

        '''

        Loading the MSCOCO 2014 data set and creating dictionaries for 
        ease of access

        '''

        # load COCO metadata
        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            self.data = json.load(f)

        # Creating the Image --> List of Captions dictionary

        self.image_to_caps = defaultdict(list)

        for caption in self.data["annotations"]:
            self.image_to_caps[caption["image_id"]].append(caption["id"])

        # Creating the Caption --> Image dictionary

        self.cap_to_image = defaultdict(None)

        for caption in self.data["annotations"]:
            self.cap_to_image[caption["id"]] = caption["image_id"]

        # Creating the Caption --> Actual Caption dictionary

        self.cap_to_cap = defaultdict(None)

        for caption in self.data["annotations"]:
            self.cap_to_cap[caption["id"]] = caption["caption"]

        # Creating List of Unique Vocabulary
        
        self.vocab = []

        for caption in self.data["annotations"]:
            text = (strip_punc(caption["caption"])).lower().split()
            for word in text:
                if word not in self.vocab:
                    self.vocab.append(word)
                else:
                    pass



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



        



        
        

        
