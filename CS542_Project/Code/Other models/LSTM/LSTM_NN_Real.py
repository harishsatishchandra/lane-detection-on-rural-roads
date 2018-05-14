import numpy as np
import h5py
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers import ConvLSTM2D, Reshape, Conv3D, MaxPooling3D, UpSampling3D, InputLayer, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

#Load images and labels
file2 = h5py.File('line28.h5','r')
data = file2['images'][:]
data2 = file2['labels'][:]
train_images = data
labels = data2
file2.close()

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
train_images = train_images[0:12750,:,:,:]
labels = np.array(labels)
labels = labels[0:12750,:,:,:]
#train_images = np.reshape(train_images,(547,21,80,160,3))

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# Using a generator to help the model use less data
# Channel shifts help with shadows slightly
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 128
epochs = 10
pool_size = (1, 2, 2)

frames = 15
X_train=np.reshape(X_train,(-1,frames,80,160,3))
y_train=np.reshape(y_train,(-1,frames,80,160,1))
X_val=np.reshape(X_val,(-1,frames,80,160,3))
y_val=np.reshape(y_val,(-1,frames,80,160,1))
input_shape = (frames, 80, 160, 3)


### Here is the actual neural network ###
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
model.add(Conv3D(8, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Conv1'))

# Conv Layer 2
model.add(Conv3D(16, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Conv2'))

# Pooling 1
model.add(MaxPooling3D(pool_size=pool_size))

# Conv Layer 3
model.add(Conv3D(16, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Conv3D(32, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Conv3D(32, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))

# Pooling 2
model.add(MaxPooling3D(pool_size=pool_size))
#model.add(Reshape(target_shape=(1,16,36,32)))

# Conv Layer 6
model.add(ConvLSTM2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu',
	recurrent_activation='hard_sigmoid', use_bias=True, bias_initializer='zeros', return_sequences = True, name = 'Conv6'))
model.add(Dropout(0.2))

# Conv Layer 7
model.add(ConvLSTM2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu',
	recurrent_activation='hard_sigmoid', use_bias=True, bias_initializer='zeros', return_sequences = True, name = 'Conv7'))
model.add(Dropout(0.2))

#model.add(Reshape(target_shape=(12,32,64)))

# Pooling 3
model.add(MaxPooling3D(pool_size=pool_size))

# Upsample 1
model.add(UpSampling3D(size=pool_size))

# Deconv 1
model.add(Conv3DTranspose(64, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Deconv1'))
model.add(Dropout(0.2))

# Deconv 2
model.add(Conv3DTranspose(64, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Deconv2'))
model.add(Dropout(0.2))

# Upsample 2
model.add(UpSampling3D(size=pool_size))

# Deconv 3
model.add(Conv3DTranspose(32, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Deconv3'))
model.add(Dropout(0.2))

# Deconv 4
model.add(Conv3DTranspose(32, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Deconv4'))
model.add(Dropout(0.2))

# Deconv 5
model.add(Conv3DTranspose(16, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Deconv5'))
model.add(Dropout(0.2))

# Upsample 3
model.add(UpSampling3D(size=pool_size))

# Deconv 6
model.add(Conv3DTranspose(16, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Deconv6'))

# Final layer - only including one channel so 1 filter
model.add(Conv3DTranspose(1, (1, 3, 3), padding='valid', strides=(1,1,1), activation = 'relu', name = 'Final'))

### End of network ###

def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 15, 80, 160, 3))
 batch_labels = np.zeros((batch_size, 15, 80,160,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.randint(0,764)
     print index
     batch_features[i] = features[index]
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')
#model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
#epochs=epochs, verbose=1, validation_data=(X_val, y_val))
#model.fit(X_train, y_train, batch_size=8, verbose=1, epochs=1, validation_data=(X_val, y_val))
model.fit_generator(generator(X_train, y_train, 8), samples_per_epoch=50, 
	nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

# Freeze layers since training is done
model.trainable = False
model.compile(optimizer='Adam', loss='mean_squared_error')

# Save model architecture and weights
model.save('full_CNN_model.h5')

# Show summary of model
model.summary()
