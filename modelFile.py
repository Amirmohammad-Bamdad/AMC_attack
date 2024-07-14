from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import layers



class LSTM(keras.Model):
    def __init__(self, input_shape, num_classes = 11):
        super(LSTM, self).__init__()

        #self.inputs = layers.Input(input_shape, name='input')
        self.lstm1 = layers.LSTM(units=128, return_sequences=True)
        self.lstm2 = layers.LSTM(units=128)
        self.outputs = layers.Dense(num_classes, activation="softmax")
    
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        out = self.outputs(x)
        return out



class VTCNN(keras.Model):
    def __init__(self, input_shape, num_classes, weights= None):

        super(VTCNN, self).__init__()
        
        self.dropout_rate = 0.5
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        # Model layers
        self.reshape = layers.Reshape(input_shape + (1,), input_shape=input_shape)
        self.conv1 = layers.Conv2D(50, (1, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform')
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.conv2 = layers.Conv2D(50, (2, 8), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform')
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1")
        self.dropout3 = layers.Dropout(self.dropout_rate)
        self.dense2 = layers.Dense(num_classes, kernel_initializer='he_normal', name="dense2")
        self.activation = layers.Activation('softmax')

        self.model = keras.Sequential([
            self.reshape,
            self.conv1,
            self.dropout1,
            self.conv2,
            self.dropout2,
            self.flatten,
            self.dense1,
            self.dropout3,
            self.dense2,
            self.activation
        ])  

        # Load weights if provided
        if weights is not None:
            self.load_weights(weights)




