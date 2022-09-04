Practice setting up building and training model using a small text first

- For each sentence, Keras expects a NumPy matrix containing one-hot vectors for each token.
- Keras will fit — or train — the seq2seq model using these matrices of one-hot vectors:
  + the encoder input data
  + the decoder input data
  + the decoder target data

Encoder training set up:
- Set up encoder requires two layer types from Keras:
  + An input layer, which defines a matrix to hold all the one-hot vectors that we’ll feed to the model.
  + An LSTM layer, with some output dimensionality (the size of the LSTM’s hidden states, which helps determine how closely the model molds itself to the training data)
- Then link the LSTM layer with our input layer (encoder outputs aren't important so discard it) then keep encoder state in a list

Decoder training set up:
- Similar to encoder training set up
- But pass in the state data from the encoder, along with the decoder inputs --> keep the output instead of the states
- Run the output through a final activation layer, using the Softmax function --> give us the probability distribution — where all probabilities sum to one — for each token
- The final layer also transforms our LSTM output from a dimensionality of whatever we gave it to the number of unique words within the hidden layer’s vocabulary (the number of unique target tokens)
(Dense layer type is the least complex for activation layer)

Train the seq2seq model:
- Keras models demand two arguments to compile:
  + An optimizer (use RMSprop here, a fancy version of gradient descent) to minimize our error rate (error at guessing the true next word given the previous words in a sentence)
  + A loss function (use logarithm-based cross-entropy function here) to determine the error rate
- Fit the model - adjust:
  + batch size (smaller batch sizes mean more time)
  + the number of epochs or cycles of training (more epochs mean a model that is more trained on the dataset -->  process will take more time)
  + validation split 
