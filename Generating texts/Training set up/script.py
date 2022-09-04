from tensorflow import keras
import numpy as np
import re
from preprocessing import input_docs, target_docs, input_tokens, target_tokens, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length

#need layers from keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

print('Number of samples:', len(input_docs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
# Build out target_features_dict:
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])


# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
# Build out reverse_target_features_dict:
reverse_target_features_dict = dict((i,token) for token, i in target_features_dict.items())

encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
print("\nHere's the first item in the encoder input matrix:\n", encoder_input_data[0], "\n\nThe number of columns should match the number of unique input tokens and the number of rows should match the maximum sequence length for input sentences.")

# Build out the decoder_input_data matrix:
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length,num_decoder_tokens), dtype='float32')
# Build out the decoder_target_data matrix:
decoder_target_data = np.zeros((len(target_docs), max_decoder_seq_length,num_decoder_tokens), dtype='float32')

for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    
  for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):

    print("Encoder input timestep & token:", timestep, token)
    # Assign 1. for the current line, timestep, & word in encoder_input_data
    encoder_input_data[line, timestep, input_features_dict[token]] = 1.

  for timestep, token in enumerate(target_doc.split()):

    # decoder_target_data is ahead of decoder_input_data by one timestep
    print("Decoder input timestep & token:", timestep, token)
    # Assign 1. for the current line, timestep, & word in decoder_input_data
    decoder_input_data[line, timestep, target_features_dict[token]] = 1.
    if timestep > 0:
      # decoder_target_data will be ahead by one timestep and will not include the start token.
      print("Decoder target timestep:", timestep)
      # Assign 1. for the current line, timestep, & word in decoder_target_data:
      decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.
        
# Create the input layer to define a matrix to hold all the one-hot vectors that we’ll feed to the model
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Create the LSTM layer with some output dimensionality
encoder_lstm = LSTM(256, return_state=True)
# Retrieve the outputs and states
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
# Put the states together in a list
encoder_states = [state_hidden, state_cell] 

# The decoder input and LSTM layers:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
# Retrieve the LSTM outputs and states:
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# Build a final Dense layer - a final activation layer, using the Softmax function, that will give us the probability distribution — where all probabilities sum to one — for each token
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# Filter outputs through the Dense layer --> transforms our LSTM output from a dimensionality to the number of unique words within the hidden layer’s vocabulary 
decoder_outputs = decoder_dense(decoder_outputs)

