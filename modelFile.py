from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import layers, models
import numpy as np

def predict_in_batches(model, input_data, batch_size= 2048):
    num_samples = input_data.shape[0]
    predictions = []

    for i in range(0, num_samples, batch_size):
        batch_data = input_data[i:i+batch_size]
        batch_predictions = model(batch_data)
        predictions.append(batch_predictions)

    return np.concatenate(predictions, axis=0)

## LSTM Model
class LSTM_AMC(keras.Model):
    def __init__(self, input_shape, num_classes = 11):
        super(LSTM_AMC, self).__init__()

        #self.inputs = layers.Input(input_shape, name='input')
        self.lstm1 = layers.LSTM(units=128, return_sequences=True)
        self.lstm2 = layers.LSTM(units=128)
        self.outputs = layers.Dense(num_classes, activation="softmax")
    
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        out = self.outputs(x)
        return out


## ResNet Model
class ResBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        self.expansion = expansion
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
        self.conv2 = layers.Conv2D(out_channels * self.expansion, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        
    def call(self, inputs):
        identity = inputs
        
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(inputs)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(tf.keras.Model):
    def __init__(self, input_shape, num_layers, num_classes=11):
        super(ResNet, self).__init__()
        if num_layers == 18:
            layers_config = [2, 2, 2, 2]
            self.expansion =1 
        
        self.in_channels = 64
        self.reshape = layers.Reshape(input_shape + (1,), input_shape=input_shape)

        self.conv1 = layers.Conv2D(self.in_channels, kernel_size=7, strides=2, padding='same', use_bias=False, input_shape=input_shape)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        
        self.layer1 = self._make_layer(64,  layers_config[0])
        self.layer2 = self._make_layer(128, layers_config[1], stride=2)
        self.layer3 = self._make_layer(256, layers_config[2], stride=2)
        self.layer4 = self._make_layer(512, layers_config[3], stride=2)
        
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

                
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = models.Sequential([
                layers.Conv2D(out_channels * self.expansion, kernel_size=1, strides=stride),
                layers.BatchNormalization()
            ])
        
        layers_list = []
        layers_list.append(ResBlock(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion
        
        for _ in range(1, blocks):
            layers_list.append(ResBlock(self.in_channels, out_channels, expansion= self.expansion))
        
        return models.Sequential(layers_list)
    
            
    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.fc(x)
        
        return x


## CNN1 Model
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
