import os.path
from gensim.models import Word2Vec, KeyedVectors

VECTOR_SIZE = 256
w2v_path = 'w2v.model'


def train_word2vec(data):
    print("\nTraining word2vec model...")
    w2v = Word2Vec(sentences=data, min_count=3, vector_size=VECTOR_SIZE)
    w2v.wv.save_word2vec_format(w2v_path, binary=False)
    w2v = KeyedVectors.load_word2vec_format(w2v_path)
    print(f'{w2v.most_similar("horrible")=}')
    print(f'{w2v.most_similar("galaxy")=}')
    print(f'{w2v.most_similar_cosmul(positive=["woman", "king"], negative=["man"])=}')
    print(f'{w2v.doesnt_match("woman king queen movie".split())=}')


def get_w2v(data):
    if not os.path.exists(w2v_path):
        train_word2vec(data)
    return KeyedVectors.load_word2vec_format(w2v_path)
