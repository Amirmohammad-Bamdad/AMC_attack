import loader
import modelFile
import keras 


epochs = 10000
batch_size = 500
path = 'E:\Clemson\Codes\DL_AMC\RML2016.10a\RML2016.10a_dict.pkl'

data = loader.RMLDataset(path)
mods = data.mods
x_train, y_train = data.train_data[0], data.train_data[1]
x_val, y_val = data.val_data[0], data.val_data[1]
x_test, y_test = data.test_data[0], data.test_data[1]


model = modelFile.CNNModel(input_shape=(2, 128), classes= len(mods)).model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


weight_path = './weights.h5'
callbacks = [
    keras.callbacks.ModelCheckpoint(weight_path, monitor='val_loss', verbose= 1, save_best_only= True, mode= 'auto'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= 0.5, verbose= 1, patince= 5, min_lr= 0.000001),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience= 50, verbose= 1, mode= 'auto')
    ]

history = model.fit(x_train, y_train, batch_size= batch_size, epochs=10,
                     validation_data=[x_val, y_val], callbacks= callbacks)

test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

