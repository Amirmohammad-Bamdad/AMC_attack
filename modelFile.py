from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import layers

class CNNModel(keras.Model):
    def __init__(self, input_shape, classes, weights= None):

        super(CNNModel, self).__init__()
        
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
        self.dense2 = layers.Dense(classes, kernel_initializer='he_normal', name="dense2")
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


    def call(self, input):
        return self.model(input)

