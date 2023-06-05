from tensorflow.keras.layers import LSTM, Dense, Conv1D
from tensorflow.keras.layers import Bidirectional, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras import Sequential


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
