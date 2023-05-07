import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import sys


def progress(j, count, prefix, size=60):
    x = int(size * j / count)
    print(f"\r{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count}", file=sys.stdout, end='')


counter = 0


def preprocess(data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text().lower()

    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    def remove_special_characters(text):
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', text)
        return text

    def lemmatize(text) -> list:
        tokens = word_tokenize(text)
        tokens = list(map(lemmatizer.lemmatize, tokens))
        return list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))

    def remove_stopwords(tokens):
        return list(filter(lambda x: not x in stop_words, tokens))

    def clean(text):
        global counter
        progress(counter, total, "Cleaning: ")
        counter += 1
        preprocessing = [
            strip_html,
            remove_between_square_brackets,
            remove_special_characters,
            lemmatize,
            remove_stopwords,
        ]
        for prep in preprocessing:
            text = prep(text)
        return text

    total = len(data)
    data = list(map(lambda x: clean(x), data))
    return data
