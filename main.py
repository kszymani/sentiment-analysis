import os.path
import pathlib
from os.path import join

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request

from plot_progress import PlotProgress, save_training_info
from preprocessing import preprocess
from word2vec import get_w2v

imdb_url = 'https://github.com/SK7here/Movie-Review-Sentiment-Analysis/raw/master/IMDB-Dataset.csv'
imdb_file = 'IMDB Dataset.csv'


def vectorize_data(data, vocab):
    print('Vectorize sentences...', end='\r')
    keys = list(vocab.keys())

    def filter_unknown(word):
        return vocab.get(word, None) is not None

    def encode(review):
        list(map(keys.index, filter(filter_unknown, review)))

    vectorized = list(map(encode, data))
    print('Vectorize sentences... (done)')
    return vectorized


def get_embedding(input_length, w2v):
    embedding_matrix = w2v.vectors
    return Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        input_length=input_length,
        weights=[embedding_matrix],
        trainable=False
    )


def get_model(embed):
    model = Sequential()
    model.add(embed)
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def callbacks():
    return [
        ModelCheckpoint(filepath='checkpoint.h5',
                        save_freq='epoch', verbose=0, monitor='val_accuracy',
                        save_weights_only=True, save_best_only=True),
        EarlyStopping(monitor="val_loss", verbose=1, patience=10, restore_best_weights=True),
        # LearningRateScheduler(scheduler),
        ReduceLROnPlateau(monitor="val_loss", factor=np.sqrt(0.1), patience=5, verbose=1, min_lr=0.5e-6),
        PlotProgress('./training_info'),
    ]


def main():
    input_length = 150
    if not os.path.exists(imdb_file):
        urllib.request.urlretrieve(imdb_url, imdb_file)
    imdb_data = pd.read_csv(imdb_file)
    print(imdb_data.describe())
    X = preprocess(imdb_data['review'])
    Y = imdb_data['sentiment'].values.tolist()
    w2v = get_w2v(X)
    X_pad = pad_sequences(
        sequences=vectorize_data(X, vocab=w2v.key_to_index),
        maxlen=input_length,
        padding='post')

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad,
        Y,
        test_size=0.2,
        shuffle=True,
        random_state=69)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        shuffle=True,
        random_state=69)

    model = get_model(get_embedding(input_length, w2v))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=3e-4),
        metrics=['accuracy', Precision(), Recall()],
    )
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=100,
        epochs=100,
        callbacks=callbacks(),
    )
    model.save('model')
    save_training_info(history, './training_info')
    print(test(model, X_test, y_test))


def test(model, X, Y):
    # model = load_model(model_path)
    testing_path = 'testing'
    pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)

    Y_pred = model.predict(X)

    plt.ion()
    pp_matrix_from_data(Y, Y_pred, columns=['negative', 'positive'])
    plt.savefig(join(testing_path, 'conf_matrix.png'))
    plt.close('all')
    plt.ioff()
    cr = classification_report(Y, Y_pred, target_names=['negative', 'positive'])
    with open(join(testing_path, "classification_scores.txt"), 'w') as f:
        print(cr, file=f)
    return accuracy_score(actual_classes, predicted_classes)


if __name__ == '__main__':
    main()
