Retrieval-based chatbots are used in closed-domain scenarios and rely on a collection of predefined responses to a user message. A retrieval-based bot completes three main tasks: intent classification, entity recognition, and response selection.

There are a number of ways to determine which response is the best fit for a given user message. One of the most important decisions a chatbot architect makes is the selection of a similarity metric.

Bag-of-Words (BoW) models are commonly used to compute intent similarity measures based on word overlap.

Term frequency–inverse document frequency (tf-idf) is another common similarity metric which incorporates the relative frequency of terms across the collection of possible responses. The sklearn package provides a TfidfVectorizer() object that we can use to fit tf-idf models.

Entity recognition tasks often extract proper nouns from a user message using Part of Speech (POS) tagging. POS tagging can be performed with nltk’s .pos_tag() method.

It’s often helpful to imagine pre-defined chatbot responses as a kind of MadLibs story. We can use word embeddings models, like the one implemented in the spacy package, to insert entities into response objects based on their cosine similarity with abstract, “blank-spot” concepts.

The final response selection relies on results from both intent classification and entity recognition tasks in order to produce a coherent response to the user message.
