import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

path = 'E:\Clemson\Codes\AMC_attack\RML2016.10a\RML2016.10a_dict.pkl'

class RMLDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.raw_data, self.mods, self.snrs = self.load_data(data_dir)

        self.train_indices, self.val_indices, self.test_indices, self.label,\
            self.train_data, self.val_data, self.test_data =\
                self.splitter(self.raw_data, self.mods, self.snrs)
        

    def load_data(self, dir):
        data = pickle.load(open(dir, 'rb'), encoding="latin1")

        modulation = []
        SNR = []

        modulation = set(key[0] for key in data.keys())
        SNR = set(key[1] for key in data.keys())

        modulation = sorted(list(modulation))
        SNR = sorted(list(SNR))

        return data, modulation, SNR

    
    def one_hotter(self, mods, labels):       
        one_hot_encoded = to_categorical(labels, num_classes=len(mods))
        return one_hot_encoded
    
    
    def splitter(self, raw_data, mods, snrs, train_ratio= 0.8, val_ratio= 0.1, test_ratio= 0.1):
        data = []
        label = []
        a = 0
        train_indices = []
        val_indices = []
        test_indices = []

        for mod in mods:
            for snr in snrs:
                data.append(raw_data[(mod,snr)])
                size = raw_data[(mod,snr)].shape[0] # Number of samples for each modulation and SNR combination
                label.extend([(mod, snr)] * size)

                train_indices += list(np.random.choice(range(a*size,(a+1)*size), size= int(train_ratio*size), replace= False))
                val_indices += list(np.random.choice(list(set(range(a*size,(a+1)*size))-set(train_indices)),
                                                      size= int(val_ratio*size), replace= False))
                
                a+=1
                
        data = np.vstack(data)
        samples_num = data.shape[0]
        
        test_indices = list(set(range(0, samples_num))-set(train_indices)-set(val_indices))

        # Shuffle
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        train_input= data[train_indices]
        val_input= data[val_indices]
        test_input=  data[test_indices]

        train_labels = [mods.index(label[idx][0]) for idx in train_indices]
        val_labels = [mods.index(label[idx][0]) for idx in val_indices]
        test_labels = [mods.index(label[idx][0]) for idx in test_indices]

        train_labels = self.one_hotter(mods, train_labels)
        val_labels = self.one_hotter(mods, val_labels)
        test_labels = self.one_hotter(mods, test_labels)  

        return train_indices, val_indices, test_indices, label,\
                    (train_input, train_labels), (val_input, val_labels), (test_input, test_labels)
