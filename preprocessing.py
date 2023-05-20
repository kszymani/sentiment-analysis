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

processed_reviews_counter = 1

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(data):
    total = len(data)
    data = list(map(lambda x: clean(x, total), data))
    global processed_reviews_counter
    processed_reviews_counter = 1
    return data


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
    return list(filter(lambda x: x not in stop_words, tokens))


def clean(text, total):
    global processed_reviews_counter
    progress(processed_reviews_counter, total, "Cleaning: ")
    processed_reviews_counter += 1
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


def progress(j, count, prefix, size=60):
    x = int(size * j / count)
    print(f"\r{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count}", file=sys.stdout, end='')
    if j == count:
        print()
