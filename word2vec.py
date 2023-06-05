from gensim.models import Word2Vec

VECTOR_SIZE = 256


def train_word2vec(data, debug=False):
    print("\nTraining word2vec model...")
    w2v = Word2Vec(sentences=data, min_count=3, vector_size=VECTOR_SIZE)

    if debug:  # Some playing around
        print(f'{w2v.wv.most_similar("horrible")=}')
        print(f'{w2v.wv.most_similar("movie")=}')
        print(f'{w2v.wv.most_similar_cosmul(positive=["woman", "king"], negative=["man"])=}')
        print(f'{w2v.wv.doesnt_match("woman king queen movie".split())=}')
    return w2v.wv
