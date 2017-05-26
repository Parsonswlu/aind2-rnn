import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = [series[i:(i+window_size)] for i in range(len(series)-window_size)]
    y = [series[i+window_size] for i in range(len(series)-window_size)]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # Initialize model as Sequential
    model = Sequential()
    # Add LSTM layer with 5 hidden units, accepting inputs of size (window_size,step_size)
    model.add(LSTM(5, input_shape = (window_size,step_size)))
    # Add Dense layer with 1 output node and linear activation function
    model.add(Dense(1, activation='linear'))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text

    # Use a list comprehension to break text into a list of characters
    # Use the uniqueness of the set object to remove duplicate values, then return to a list
    unique_chars = list(set([i for i in text]))

    # remove as many non-english characters and character sequences as you can 

    # define non-english characters as anything that's not lowercase ASCII characters 
    # or standard punctuation (space, exclimation mark, comma, period, colon, semi-colon, question mark)
    import string
    all_chars = string.ascii_lowercase
    all_punctuation = [' ', '!', ',', '.', ':', ';', '?']
    # Use a list comprehension to check if each unique character is a non-english character 
    # and keep unique non-english characters in a list using set+list as above
    nonenglish_chars = list(set([i for i in unique_chars if i not in all_chars and i not in all_punctuation]))
    # For loop over each unique non-english character and replace with a space 
    for i in nonenglish_chars:
        text = text.replace(i,' ')
        
    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    # Define range of outputs
    output_range = range(0,len(text)-window_size,step_size)
    # Use a list comprehension to collect ceil((len(text)-window_size)/step_size) inputs, each of size window_size
    inputs = [text[i:(i+window_size)] for i in output_range]
    # Use a list comprehension to collect ceil((len(text)-window_size)/step_size) outputs, each of size 1    
    outputs = [text[i+window_size] for i in output_range]
    
    return inputs,outputs
