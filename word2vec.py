import os.path
from gensim.models import Word2Vec, KeyedVectors

VECTOR_SIZE = 256


def train_word2vec(data, name, debug=False):
    print("\nTraining word2vec model...")
    w2v = Word2Vec(sentences=data, min_count=3, vector_size=VECTOR_SIZE)
    w2v.wv.save_word2vec_format(name, binary=False)

    if debug:  # Some playing around
        print(f'{w2v.wv.most_similar("horrible")=}')
        print(f'{w2v.wv.most_similar("movie")=}')
        print(f'{w2v.wv.most_similar_cosmul(positive=["woman", "king"], negative=["man"])=}')
        print(f'{w2v.wv.doesnt_match("woman king queen movie".split())=}')
    return w2v.wv


def get_w2v(data, name):
    if not os.path.exists(name):
        return train_word2vec(data, name)
    return KeyedVectors.load_word2vec_format(name)
