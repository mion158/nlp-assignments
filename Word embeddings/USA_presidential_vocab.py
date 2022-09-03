import os
import gensim
import spacy

from nltk.tokenize import PunktSentenceTokenizer
from collections import Counter
# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
print(files)

# read each speech file
def read_file(file_name):
  with open(file_name, 'r+', encoding='utf-8') as file:
    file_text = file.read()
  return file_text

speeches = [read_file(speech) for speech in files]

# preprocess each speech
def process_speeches(speeches):
  word_tokenized_speeches = list()
  for speech in speeches:
    sentence_tokenizer = PunktSentenceTokenizer()
    sentence_tokenized_speech = sentence_tokenizer.tokenize(speech)
    word_tokenized_sentences = list()
    for sentence in sentence_tokenized_speech:
      word_tokenized_sentence = [word.lower().strip('.').strip('?').strip('!') for word in sentence.replace(",","").replace("-"," ").replace(":","").split()]
      word_tokenized_sentences.append(word_tokenized_sentence)
    word_tokenized_speeches.append(word_tokenized_sentences)
  return word_tokenized_speeches

tokenized_speeches = process_speeches(speeches)

# merge speeches
def merge_speeches(speeches):
  all_sentences = list()
  for speech in speeches:
    for sentence in speech:
      all_sentences.append(sentence)
  return all_sentences

all_sentences = merge_speeches(tokenized_speeches)

# view most frequently used words
def most_frequent_words(list_of_sentences):
  all_words = [word for sentence in list_of_sentences for word in sentence]
  return Counter(all_words).most_common()

most_frequent_words = most_frequent_words(all_sentences)
print(most_frequent_words)
# create gensim model of all speeches

all_presidents_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)
# view words similar to freedom
similar_to_freedom = all_presidents_embeddings.most_similar('freedom',topn=20)
print(similar_to_freedom)
