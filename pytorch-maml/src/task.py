import os
import random
import numpy as np
import torch

class FRTask(object):
    '''
    Sample a few-shot learning task from the FR dataset
    Sample N-way k-shot train and val sets according to
     - split (dataset/meta level train or test)
     - N-way classification (sample this many chars)
     - k-shot (sample this many examples from each char class)
    Assuming that the validation set is the same size as the train set!
    '''

    def __init__(self, root, num_cls, num_inst, split='train'):
        self.dataset = 'fr'
        self.root = '{}/images_background'.format(root) if split == 'train' else '{}/images_evaluation'.format(root)
        self.num_cl = num_cls
        self.num_inst = num_inst
        # Sample num_cls characters and num_inst instances of each
        galaxies = os.listdir(self.root)
        g_type = []
        for g in galaxies:
            g_type += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]
        random.shuffle(g_type)
        classes = g_type[:num_cls]
        labels = np.array(list(range(len(classes))))
        labels = dict(list(zip(classes, labels))) 
        instances = dict()
        # Now sample from the chosen classes to create class-balanced train and val sets
        self.train_ids = []
        self.val_ids = []
        for c in classes:
            # First get all isntances of that class
            temp = [os.path.join(c, x) for x in os.listdir(os.path.join(self.root, c))]
            instances[c] = random.sample(temp, len(temp))
            # Sample num_inst instances randomly each for train and val
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:num_inst*2]
        # Keep instances separated by class for class-balanced mini-batches
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]
        

    def get_class(self, instance):
        return os.path.join(*instance.split('/')[:-1])
