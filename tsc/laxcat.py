# LAXCAT model
#import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


class Classifier_LAXCAT:
    
    def __init__(self, params, input_shape, \
                 nclasses=2, verbose=False, build=True):

        self.input_shape = input_shape
        self.nclasses = nclasses
        self.nfeatures = input_shape[0]
        self.ntimesteps = input_shape[1]
        print(str(params))
        print(nclasses)
        self.nfilters = params['nfilters']
        self.kernel_size = params['kernel_size']

        if build == True:
            self.model = self.build_model()
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose


    def build_model(self):
        # input layer
        x = layers.Input(shape=self.input_shape)

        # convolution layer
        x = layers.Conv1D(self.nfilters, self.kernel_size, \
                          strides=1, padding="same")(x)

        # variable attention layer
        x = layers.Dense(self.nfilters)(x)
        x = layers.Dense(self.nfeatures, \
                         activation="softmax")(x)

        # temporal attention layer
        x = layers.Dense(self.nfilters)(x)
        x = layers.Dense(self.ntimesteps, \
                         activation="softmax")(x)

        output_layer = layers.Dense(self.nclasses, \
                                    activation='softmax')(x)

        model = tf.keras.models.Model(inputs=x, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', \
                      optimizer=tf.keras.optimizers.Adam(), \
                      metrics=['accuracy'])

        model.summary()
        return model
        
        
    def build_model0(self):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        # Convolutional layer to extract features
        model.add(layers.Conv1D(self.nfilters, self.kernel_size))

        # P = input_shape[0] (number of features in MTS)
        # J = nfilters (number of time domain features extracted with convolution)

        # Variable Attention Module
        model.add(layers.Dense(self.nfilters))
        model.add(layers.Dense(self.nfeatures))
        model.add(layers.Softmax())


        # Temporal Attention Module
        model.add(layers.Dense(self.nfilters))
        model.add(layers.Dense(self.ntimesteps))
        model.add(layers.Softmax())

        # Output layer
        model.add(layers.Dense(self.nclasses))
        model.add(tf.layer.Softmax())

        model.compile(loss='categorical_crossentropy', \
                      optimizer=tf.keras.optimizers.Adam(), \
                      metrics=['accuracy'])

        return model

        
