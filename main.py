import loader
import modelFile
import keras 
import tensorflow as tf
import os
import pickle
import utils
from sklearn.metrics import confusion_matrix
import numpy as np
import json

###########################################################################
model_name = "VTCNN"

epochs = 100
batch_size = 256
path = 'E:\Clemson\Codes\AMC_attack\RML2016.10a\RML2016.10a_dict.pkl'

data = loader.RMLDataset(path)
mods = data.mods
snrs = data.snrs
labels = data.label
test_indices = data.test_indices
model = modelFile.VTCNN(input_shape=(2, 128), num_classes= len(mods)).model
#model = modelFile.LSTM_AMC(input_shape=(128, 2), num_classes= len(mods))
###########################################################################

x_train, y_train = data.train_data[0], data.train_data[1]
x_val, y_val = data.val_data[0], data.val_data[1]
x_test, y_test = data.test_data[0], data.test_data[1]

weight_path = f'./{model_name}_weights/'
callbacks = [
    keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose= 1, save_best_only= True, mode= 'auto', save_format="tf"),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= 0.5, verbose= 1, patince= 5, min_lr= 0.000001),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience= 50, verbose= 1, mode= 'auto')
    ]


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.built = True
model.summary()

if os.path.isdir(weight_path):
    #model.load_weights(weight_path)
    model = tf.saved_model.load(weight_path)
    with open(f'{model_name}_training_history.pkl', 'rb') as f:
        history = pickle.load(f)

else:
    history = model.fit(x_train, y_train, batch_size= batch_size, epochs= epochs,
                     validation_data= [x_val, y_val], callbacks= callbacks)
    model.summary()

    with open(f'{model_name}_training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    history = history.history


test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size= batch_size)
print("Test accuracy: ", test_acc)
print("Test loss: ", test_loss)

y_test_hat = model.predict(x_test)
total_y_test = np.argmax(y_test, axis=1) #Labels
total_y_test_hat = np.argmax(y_test_hat, axis=1) #Predictions

total_cm = confusion_matrix(total_y_test, total_y_test_hat)

utils.plot_confusion_matrix(cm= total_cm, classes= mods, 
                            title=f"{model_name} Overall final confusion matrix",
                            save_filename= f"figure/{model_name}_Overall_final_confusion_matrix.png")

utils.total_plotter(history, model_name)

if os.path.isfile('acc_mod_snr.json') and os.path.isfile('acc.json') and os.path.isfile('bers.json'):
    with open('acc_mod_snr.json', 'r') as f:  
        acc_mod_snr = json.load(f)['acc_mod_snr']
    with open('acc.json', 'r') as f:  
        acc = json.load(f)['acc']
    with open('bers.json', 'r') as f:  
        bers = json.load(f)['bers']

else:
    acc, acc_mod_snr, bers = utils.evaluate_per_snr(model= model, X_test= x_test, Y_test= y_test,
                                           snrs= snrs, classes= mods, labels= labels,
                                             test_indices= test_indices, model_name= model_name)

    utils.save_results(acc= acc, acc_mod_snr= acc_mod_snr, bers= bers, model_name= model_name)

utils.plot_accuracy_per_snr(snrs= snrs, acc_mod_snr= acc_mod_snr, classes= mods, model_name= model_name)
utils.plot_ber_vs_snr(snrs, bers, model_name)