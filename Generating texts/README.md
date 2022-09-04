Has two types:

Type 1: build and train the model with target sentence sequence --> teacher forcing method to train seq2seq decoders

Type 2: test the model without target sentence sequence --> need to decode step by step

- seq2seq models are deep learning models that use recurrent neural networks like LSTMs to generate output.
- Seq2seq networks encompass two main parts:
  + The encoder accepts language as input and outputs state vectors.
  + The decoder accepts the encoder’s final state and outputs possible translations.
- Need to mark the beginning "<START>" and end "<END>" of target sentences so that the decoder knows what to expect at the beginning and end of sentences.
- One-hot vectors are a way to represent a given word in a set of words wherein we use 1 to indicate the current word and 0 to indicate every other word.
- Timesteps help us keep track of where we are in a sentence.
- Can adjust batch size that determines how many sentences we give a model at a time, and tweak dimensionality and number of epochs, which can improve results with careful tuning.
- The Softmax function converts the output of the LSTMs into a probability distribution over words in our vocabulary.
