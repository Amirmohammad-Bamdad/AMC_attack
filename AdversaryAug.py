from sklearn.model_selection import train_test_split
import numpy as np
import pickle 

class datasetAug:
    def __init__ (self, model_name, model, x_train, y_train, x_val, y_val):
        self.data_train = x_train.copy()
        self.labels_train = y_train.copy()
        
        self.data_val = x_val.copy()
        self.labels_val = y_val.copy()
        
        self.model = model
        self.model_name = model_name


    def sampler(self):
        x_sampled_train, _, y_sampled_train, _ =\
            train_test_split(self.data_train, self.labels_train, test_size=0.0001, random_state=42)
        
        x_sampled_val, _, y_sampled_val, _ =\
            train_test_split(self.data_val, self.labels_val, test_size=0.0001, random_state=42)
        
        return x_sampled_train, y_sampled_train, x_sampled_val, y_sampled_val

    
    def augment_dataset(self, adv_x_train, y_sampled_train,
                         adv_x_val, y_sampled_val):
        
        x_train = np.concatenate([self.data_train, adv_x_train])
        y_train = np.concatenate([self.labels_train, y_sampled_train])
        
        x_val = np.concatenate([self.data_val, adv_x_val])
        y_val = np.concatenate([self.labels_val, y_sampled_val])

        return x_train, y_train, x_val, y_val
    

    def adversary_train(self, batch_size, epochs, aug_x_train, aug_y_train,
                         aug_x_val, aug_y_val, callbacks):
        
        print("\nStart of Adversarial Training ...")
        print(aug_x_train.shape)
        print(aug_y_train.shape)
        history = self.model.fit(aug_x_train, aug_y_train, batch_size= batch_size, epochs= epochs,
                 validation_data= [aug_x_val, aug_y_val], callbacks= callbacks)
        
        with open(f'./history_and_metrics/adv_{self.model_name}_training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
