import os.path
import pathlib
from os.path import join

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, MaxPooling1D, GlobalAveragePooling1D
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from scipy.stats import ttest_rel
from sklearn.metrics import classification_report, accuracy_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv1D, Flatten
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import binary_crossentropy

import datasets
from datasets import get_data
from plot_progress import PlotProgress, save_training_info

import warnings

warnings.filterwarnings('ignore')

batch_size = 128
max_epochs = 100

scores_file = "test_scores.npy"


def main():
    dataset_paths = [
        datasets.multi_domain,
        datasets.imdb_file,
        datasets.twitter_train_file,
    ]
    models = [
        get_model_cnn_lstm,
        get_model_lstm_cnn
    ]
    n_splits = 5
    scores = np.zeros((len(dataset_paths), len(models), n_splits))

    for d_id, ds in enumerate(dataset_paths):
        (X, y), (X_test, y_test), embed = get_data(ds)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2137)
        for fold_id, (train, valid) in enumerate(skf.split(X, y)):
            for m_id, model_type in enumerate(models):
                model = model_type(embed)
                tr = (X[train], y[train])
                val = (X[valid], y[valid])
                ds_name = ds.split(r'/')[1].rstrip('.csv')
                model_name = f'{ds_name}-f{fold_id}-m{m_id}'
                if not os.path.exists(os.path.join(model_name, 'saved_model.pb')):
                    train_model(model, tr, val, model_name=model_name)
                scores[d_id, m_id, fold_id] = test(model, X_test, y_test, test_name=model_name)
        np.save(scores_file, scores)
        run_stats()


def train_model(model, train, val, model_name):
    pathlib.Path(model_name).mkdir(exist_ok=True, parents=True)
    X_train, y_train = train
    X_val, y_val = val
    model.compile(
        **compile_opts()
    )
    try:
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=callbacks(model_name),
        )
    except KeyboardInterrupt:
        print("\nInterrupted!")
        weights_path = join(model_name, 'checkpoint.h5')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
    model.save(model_name)
    save_training_info(model.history, f'{model_name}/training_info')


def test(model, X, Y, test_name):
    testing_path = f'{test_name}/testing'
    pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)

    Y_pred = model.predict(X)
    Y_pred_classes = np.round(Y_pred)

    plt.ion()
    pp_matrix_from_data(Y, Y_pred_classes, columns=['negative', 'positive'])
    plt.savefig(join(testing_path, 'conf_matrix.png'))
    plt.close('all')
    plt.ioff()

    RocCurveDisplay.from_predictions(Y, Y_pred)
    plt.savefig(join(testing_path, 'roc.png'))
    plt.close('all')

    cr = classification_report(Y, Y_pred, target_names=['negative', 'positive'])
    with open(join(testing_path, "classification_scores.txt"), 'w') as f:
        print(cr, file=f)
    return accuracy_score(Y, Y_pred)


def run_stats(alfa=0.05):
    scores = np.load(scores_file)
    n_models = scores.shape[1]
    for ds_scores in scores:
        t_statistic = np.zeros((n_models, n_models))
        p_value = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                t_statistic[i, j], p_value[i, j] = ttest_rel(ds_scores[i], ds_scores[j])
        np.save(f't_stat.npy', t_statistic)
        print(f'{t_statistic=}')
        print(f'{p_value=}')
        np.save(f'p_stat.npy', p_value)
        advantage = np.zeros((n_models, n_models))
        advantage[t_statistic > 0] = 1
        significance = np.zeros((n_models, n_models))
        significance[p_value <= alfa] = 1
        stat_better = significance * advantage
        print(stat_better)


def get_model_lstm_cnn(embed):
    model = Sequential()
    model.add(embed)
    model.add(Bidirectional(LSTM(48, recurrent_dropout=0.1, return_sequences=True)))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def get_model_cnn_lstm(embed):
    model = Sequential()
    model.add(embed)
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(48, recurrent_dropout=0.1)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def callbacks(test_name):
    return [
        ModelCheckpoint(filepath=f'{test_name}/checkpoint.h5',
                        save_freq='epoch', verbose=0, monitor='val_accuracy',
                        save_weights_only=True, save_best_only=True),
        EarlyStopping(monitor="val_loss", verbose=1, patience=10, restore_best_weights=True),
        # LearningRateScheduler(scheduler),
        ReduceLROnPlateau(monitor="val_loss", factor=np.sqrt(0.1), patience=5, verbose=1, min_lr=0.5e-6),
        PlotProgress(f'{test_name}/training_info'),
    ]


def compile_opts():
    return {
        'loss': binary_crossentropy,
        'optimizer': Adam(learning_rate=3e-4),
        'metrics': ['accuracy', Precision(), Recall()]
    }


if __name__ == '__main__':
    main()
