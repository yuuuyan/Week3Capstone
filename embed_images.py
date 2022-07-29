import numpy as np
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import iterate_structure
from mygrad.nnet.losses import margin_ranking_loss
import database as db
import embed_captions as ec
import organizeCOCO as oc
import random 
import mygrad as mg
import mynn
from cogworks_data.language import get_data_path
from pathlib import Path
from mygrad.nnet.initializers import glorot_normal
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam
import pickle

with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)

class Model:
    def __init__(self):
        """ Initializes layers in  model, and sets them
        as attributes of model.
        """
        self.w_embed = dense(512, 200, weight_initializer=glorot_normal, bias=False)
        
        
    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, 512)
            An array of image descriptor vectors
        Returns
        -------
        mygrad.Tensor, shape=(200,)
            The model's embedded image vectors
        '''

        w = self.w_embed(x) # (1, 200) vector
        return w / mg.sqrt(mg.sum(w))

        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        # STUDENT CODE HERE
        return self.w_embed.parameters

    #add save and load model functions using np.save
    def save_weights(self):
        np.save("weights.npy", self.w_embed)
    
    def load_weights(self):
        self.w_embed = np.load("weights.npy", self.w_embed)
    

def train_model():
    '''
    Creates model instance
    Get a plethora of images and split up into true and confusor images
    Get caption embedding for true image
    Get image embedding for true and confusor images
    Do the wacky comparison function thingy 
    Optimize :)
    return model
    '''
    model = Model() # initialize model

    optim = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)# initialize optimizer

    num_epochs = 6
    batch_size = 32
    
    len_coco = 82612
    coco_data = oc.COCO()

    losses = []
    loss_cnt = 0

    for epoch_cnt in range(num_epochs):
        # Use np.random.shuffle(idxs) to shuffle the indices in-place 
        train_idxs = np.arange(0, len_coco // 5 * 4)# create array of N indices - one for each datum in the dataset
        test_idxs = np.arange(len_coco // 5 * 4)
        np.random.shuffle(train_idxs)
        np.random.shuffle(test_idxs)

        for batch_cnt in range(0, len(train_idxs)//batch_size):
            ids = train_idxs[batch_cnt * batch_size:(batch_cnt + 1) * batch_size] # get batch of indices
            batch = [coco_data.data["images"][i]["id"] for i in ids] # get random sample of images from COCO data

            conf_ids = [coco_data.data["images"][random.randint(0, len_coco)]["id"] for i in range(batch_size)]
            
            true_desc = []
            conf_desc = []
            captions = []
            
            for i in range(batch_size):
                if batch[i] in resnet18_features and conf_ids[i] in resnet18_features and batch[i] in coco_data.image_to_caps:
                    true_desc.append(resnet18_features[batch[i]])
                    conf_desc.append(resnet18_features[conf_ids[i]])

                    # all_captions = [coco_data.I_To_C(i) for i in batch]
                    captions.append(coco_data.rand_cap(batch[i]))

            # true_desc = true_desc[~np.all(true_desc == 0, axis=0)]

            true_embed = model(np.array(true_desc))
            conf_embed = model(np.array(conf_desc))
            
            
            caption_embed = np.array([ec.embed(c) for c in captions])
        
            '''
            for true_id in batch:
                conf_id = coco_data.data["id"][random.randint(0, len_coco)]

                true_desc = resnet18_features[true_id] 
                conf_desc = resnet18_features[conf_id]

                all_captions = coco_data.I_To_C[true_id]
                caption = all_captions[random.randint(0, len(all_captions))]

                true_embed = model(true_desc)
                conf_embed = model(conf_desc)
                caption_embed = ec.embed(caption)

                sim_true.append(true_embed @ caption_embed)
                sim_conf.append(conf_embed @ caption_embed)
            '''

            # margin_ranking_loss(x1, x2, y, margin) equivalent to mg.mean(mg.maximum(0, margin - y * (x1 - x2)))
            caption_embed = mg.tensor(caption_embed)
            
            true_embed = true_embed.reshape((true_embed.shape[0], 200))
            conf_embed = conf_embed.reshape((conf_embed.shape[0], 200))

            sim_true = mg.einsum("nd, nd -> n", caption_embed, true_embed)
            sim_conf = mg.einsum("nd, nd -> n", caption_embed, conf_embed)
            loss = margin_ranking_loss(sim_true, sim_conf, y=1, margin=0.25)
            
            if loss_cnt % 100 == 0:
                losses.append(loss)
            
            loss.backward()
            optim.step()
            
            accuracy = np.mean(sim_true.data > sim_conf.data)

        """
        with mg.no_autodiff:
            prediction = model(iris)
            truth = iris
            loss = mean_squared_loss(prediction, truth)
        print(f'epoch {epoch_cnt:5}, loss = {loss.item():0.3f}')
        plotter.set_train_epoch()
        """
        

    '''
    for epoch_cnt in range(num_epochs):
        # Use np.random.shuffle(idxs) to shuffle the indices in-place 
        train_idxs = np.arange(0, len_coco // 5 * 4)# create array of N indices - one for each datum in the dataset
        test_idxs = np.arange(len_coco // 5 * 4)
        np.random.shuffle(train_idxs)
        np.random.shuffle(test_idxs)

        for batch_cnt in range(0, len(train_idxs)//batch_size):
            ids = train_idxs[batch_cnt * batch_size:(batch_cnt + 1) * batch_size]# get batch of indices
            batch = [coco_data.data["images"][i]["id"] for i in ids] # get random sample of images from COCO data
            
            conf_ids = [coco_data.data["id"][random.randint(0, len_coco)] for i in range(batch_size)]

            true_desc = [resnet18_features[i] for i in batch]
            conf_desc = [resnet18_features[i] for i in conf_ids]

            all_captions = coco_data.I_To_C[batch]
            caption = all_captions[random.randint(0, len(all_captions))] # randomly selects a caption for the true image

            true_embed = model(true_desc)
            conf_embed = model(conf_desc)
            caption_embed = ec.embed(caption)

            # TODO : values in sim_true and sim_conf need to be cosine distance-d
            sim_true = []
            sim_conf = []

            
            """"""
            for true_id in batch:
                conf_id = coco_data.data["id"][random.randint(0, len_coco)]

                true_desc = resnet18_features[true_id] 
                conf_desc = resnet18_features[conf_id]

                all_captions = coco_data.I_To_C[true_id]
                caption = all_captions[random.randint(0, len(all_captions))]

                true_embed = model(true_desc)
                conf_embed = model(conf_desc)
                caption_embed = ec.embed(caption)
                
                sim_true.append(true_embed @ caption_embed)
                sim_conf.append(conf_embed @ caption_embed)
            """"""
            
            # margin_ranking_loss(x1, x2, y, margin) equivalent to mg.mean(mg.maximum(0, margin - y * (x1 - x2)))
            # sim_true = np.array(sim_true)
            sim_true = mg.einsum("nd, nd -> n", caption_embed, true_embed)
            #sim_conf = np.array(sim_conf)
            sim_conf = mg.einsum("nd, nd -> n", caption_embed, conf_embed)
            loss = margin_ranking_loss(sim_true, sim_conf, y=1, margin=0.25)
            loss.backward()
            optim.step()

    return model
    '''

                
