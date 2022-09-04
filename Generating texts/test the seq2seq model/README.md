This time don't know the target sequence


To set up for testing the model:
 - build an encoder model with our encoder inputs and the placeholders for the encoder’s output states
 - build placeholders for the decoder’s input states, which we can build as input layers and store together (because don’t know what we want to decode yet or what hidden state we’re going to end up with, so we need to do everything step-by-step)
 - pass the encoder’s final hidden state to the decoder, sample a token, and get the updated hidden state back. 
 - pass the updated hidden state back into the network
 - use the decoder LSTM and decoder dense layer (with the activation function) that we trained earlier --> new decoder states and outputs
 - set up the decoder model with:
   + the decoder inputs (the decoder input layer)
   + the decoder input states (the final states from the encoder)
   + the decoder outputs (the NumPy matrix we get from the final output layer of the decoder)
   + the decoder output states (the memory throughout the network from one word to the next) 
 
 
Test the model by building a function that:
- accepts a NumPy matrix representing the test English sentence input
- uses the encoder and decoder we’ve created to generate Spanish output
Translation:
- decode the sentence word by word using the output state that we retrieved from the encoder (which becomes our decoder’s initial hidden state) --> use while loop run until either current token is "<END>" or decoded sentence hit maximum target length
 + beginning with the "<START>" token and the current state (initially passed to us from the encoder model) to get a bunch of possible next words and their corresponding probabilities
 + use NumPy’s .argmax() method to determine the token (word) with the highest probability and add it to the decoded sentence
- update the decoder hidden state after each word so that we use previously decoded words to help decode new ones


