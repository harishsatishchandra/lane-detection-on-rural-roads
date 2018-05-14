This folder contains the code for my models using ConvLSTM3D layers

Based on talking to the TA, we need to utilize the extra 'time' dimension for convLSTM2D to get anything out of it. He also said it’s probably most effective as the last few layers of our model on the convolution side. Therefore, I am going to build off of the first model I made, and go with the default parameters given in the documentation:
https://keras.io/layers/recurrent/

In building the models, feel free to test different epoch values

Models:
1. Original but replace last two conv2D layers with convLSTM2D (incorrectly)
2. All conv2D layers replaced with convLSTM2D (incorrectly, and not recommended)
3. Model 1, but done correctly (time slices) with Conv3D to pass the time layer to the LSTM layers, without convolution in the time dimension.

Note: The third model has not been run successfully. We encounter Bus Error 10, I believe resulting from overloading the computers memory.

All models run with the data ‘line28.h5’ which is the ordered data set of our original data set. Simple running ‘pyhton <filename>’ will compile and train the model, provided ‘line28.h5’ is in the same directory.
