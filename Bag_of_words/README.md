Bag-of-words (BoW) — also referred to as the unigram model — is a statistical language model based on word count.

BoW can be implemented as a Python dictionary with each key set to a word and each value set to the number of times that word appears in a text.

For BoW, training data is the text that is used to build a BoW model.

BoW test data is the new text that is converted to a BoW vector using a trained features dictionary.

A feature vector is a numeric depiction of an item’s salient features.

Feature extraction (or vectorization) is the process of turning text into a BoW vector.

A features dictionary is a mapping of each unique word in the training data to a unique index. This is used to build out BoW vectors.

BoW has less data sparsity than other statistical models. It also suffers less from overfitting.

BoW has higher perplexity than other models, making it less ideal for language prediction.

One solution to overfitting is language smoothing, in which a bit of probability is taken from known words and allotted to unknown words.
