import pathlib
from contextlib import redirect_stdout

import numpy as np
from PIL import ImageFont
from matplotlib import colors
from tensorflow.keras.layers import LSTM, Dense, Conv1D
from tensorflow.keras.layers import Bidirectional, MaxPooling1D, GlobalAveragePooling1D, Embedding
from tensorflow.keras import Sequential

import visualkeras
from tensorflow.keras import utils
from collections import defaultdict


def get_model_lstm_cnn_bi(embed):
    model = Sequential()
    model.add(embed)
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def get_model_lstm_cnn(embed):
    model = Sequential()
    model.add(embed)
    model.add(LSTM(64, recurrent_dropout=0.1, return_sequences=True))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def get_model_cnn_lstm_bi(embed):
    model = Sequential()
    model.add(embed)
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def get_model_cnn_lstm(embed):
    model = Sequential()
    model.add(embed)
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, recurrent_dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def plot_model(folder, model, filename='model'):
    layers = [
        Dense, Conv1D, Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, MaxPooling1D
    ]

    np.random.seed(1234)
    pallette = np.random.choice(list(colors.CSS4_COLORS), len(layers))
    color_map = defaultdict(dict)
    for i, l in enumerate(layers):
        color_map[l]['fill'] = pallette[i]

    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
    utils.plot_model(model, show_layer_names=False, show_shapes=True, to_file=f'{folder}/{filename}.png')
    with open(f'{folder}/{filename}.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    font = ImageFont.truetype("arial.ttf", 20)
    visualkeras.layered_view(model, to_file=f'{folder}/{filename}_layered.png', legend=True, color_map=color_map,
                             max_xy=650, scale_z=2, font=font)
