from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))
stops_add = set(["'d", "'ll", "'m", "'re", "'s", "'t", "n't", "'ve"])
stop_words = stop_words.union(stops_add)
punctuations = set(string.punctuation)
punctuations.add("''")
punctuations.add("``")
stops_and_punctuations = punctuations.union(stop_words)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def remove_stopword(list_of_words):
    return [i for i in list_of_words if i not in stops_and_punctuations]

def sentence_to_wordlist( sentence, remove_stopwords=False ):

    sentence_text = BeautifulSoup(sentence,"html5lib").get_text()

    sentence_text = re.sub("[^a-zA-Z]"," ", sentence_text)

    words = sentence_text.lower().split()

    if remove_stopwords:
        words = remove_stopword(words)

    return(words)


def document_to_sentences( document, tokenizer, remove_stopwords=False ):

    raw_sentences = tokenizer.tokenize(document.strip())

    sentences = []
    for raw_sentence in raw_sentences:

        if len(raw_sentence) > 0:

            sentences.append( sentence_to_wordlist( raw_sentence, remove_stopwords ))

    return raw_sentences,sentences

def get_word_from_document(document):

    # data = read_from_file(path_of_file)
    data = document
    sentences , list_of_sentences = document_to_sentences(data,tokenizer,remove_stopwords=True)
    return sentences,list_of_sentences
