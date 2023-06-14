import os.path
import pathlib
import shutil
from os.path import join
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import classification_report, RocCurveDisplay, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import binary_crossentropy

import datasets
from datasets import get_data, get_embedding, pad
from model import get_model_cnn_lstm, get_model_lstm_cnn, get_model_cnn_lstm_bi, get_model_lstm_cnn_bi, plot_model
from plot_progress import PlotProgress, save_training_info

import warnings

from stats import run_stats, plot_training_metrics
from word2vec import train_word2vec

warnings.filterwarnings('ignore')

batch_size = 128
max_epochs = 10

N_REPEATS = 5
N_SPLITS = 2
kfold = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)

dir_name = 'models'

models = {
    "CNN-LSTM": get_model_cnn_lstm,
    "LSTM-CNN": get_model_lstm_cnn,
    "CNN-LSTM-BI": get_model_cnn_lstm_bi,
    "LSTM-BI-CNN": get_model_lstm_cnn_bi,
    "CNN-LSTM-W2V_CONCAT": get_model_cnn_lstm,
    "LSTM-CNN-W2V_CONCAT": get_model_lstm_cnn,
}


def main():
    dataset_paths = [
        datasets.twitter_train_file,
        datasets.imdb_file,
    ]
    for d_id, ds in enumerate(dataset_paths):
        ds_name = ds.split(r'/')[1].rstrip('.csv')
        scores = np.zeros((len(models), kfold.get_n_splits()))
        scores_file = f'{dir_name}/{ds_name}/scores.npy'
        if not os.path.exists(scores_file):
            (X, y), (X_test, y_test) = get_data(ds)
            for fold_id, (train, valid) in enumerate(kfold.split(X, y)):
                X_train, y_train, X_val, y_val = X[train], y[train], X[valid], y[valid]
                for m_id, (model_name, get_model) in enumerate(models.items()):
                    model_path = f'{dir_name}/{ds_name}/{model_name.lower()}/f{fold_id}'
                    print(f'Training {model_path}')
                    pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
                    w2v_path = f'{model_path}/w2v'
                    if not os.path.exists(w2v_path):
                        if 'W2V_CONCAT' in model_name:
                            w2v = train_word2vec(np.concatenate([X_train, X_val]))
                        else:
                            w2v = train_word2vec(X_train)
                        w2v.save_word2vec_format(w2v_path, binary=False)
                    else:
                        w2v = KeyedVectors.load_word2vec_format(w2v_path)
                    if not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                        embed = get_embedding(w2v)
                        model = get_model(embed)
                        X_pad_train, X_pad_val, = pad(X_train, w2v), pad(X_val, w2v),
                        train_model(model, X_pad_train, y_train, X_pad_val, y_val, model_path=model_path)
                    else:
                        print(f'Skipping training')

            for fold_id in range(kfold.get_n_splits()):
                for m_id, (model_name, _) in enumerate(models.items()):
                    model_path = f'{dir_name}/{ds_name}/{model_name.lower()}/f{fold_id}'
                    w2v_path = f'{model_path}/w2v'
                    w2v = KeyedVectors.load_word2vec_format(w2v_path)
                    X_pad_test = pad(X_test, w2v)
                    model = load_model(model_path)
                    scores[m_id, fold_id] = test(model, X_pad_test, y_test, test_name=model_path)
            np.save(scores_file, scores)
        ds_path = f'{dir_name}/{ds_name}'
        plot_training_metrics(ds_path, metric='accuracy')
        plot_training_metrics(ds_path, metric='val_accuracy')
        scores = np.load(scores_file)
        run_stats(ds_path, scores, models=models)


def train_model(model, X_train, y_train, X_val, y_val, model_path):
    pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
    model.compile(**compile_opts())
    plot_model(f'{model_path}/model-info', model, )
    try:
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=callbacks(model_path),
        )
    except KeyboardInterrupt:
        print("\nInterrupted!")
    weights_path = join(model_path, 'checkpoint.h5')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    model.save(model_path)
    save_training_info(model.history, f'{model_path}/training_info')


def test(model, X, Y, test_name):
    testing_path = f'{test_name}/testing'
    pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)

    Y_pred = model.predict(X)
    Y_pred_classes = np.round(Y_pred).astype(np.int)

    plt.ion()
    pp_matrix_from_data(Y, Y_pred_classes, columns=['negative', 'positive'])
    plt.savefig(join(testing_path, 'conf_matrix.png'))
    plt.close('all')
    plt.ioff()

    RocCurveDisplay.from_predictions(Y, Y_pred)
    plt.savefig(join(testing_path, 'roc.png'))
    plt.close('all')

    cr = classification_report(Y, Y_pred_classes, target_names=['negative', 'positive'])
    with open(join(testing_path, "classification_scores.txt"), 'w') as f:
        print(cr, file=f)
    return accuracy_score(Y, Y_pred_classes)


def callbacks(test_name):
    monitor = 'val_loss'
    return [
        ModelCheckpoint(filepath=f'{test_name}/checkpoint.h5',
                        save_freq='epoch', verbose=0, monitor=monitor,
                        save_weights_only=True, save_best_only=True),
        EarlyStopping(monitor=monitor, verbose=1, patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor=monitor, factor=np.sqrt(0.1), patience=2, verbose=1, min_lr=0.5e-6),
        PlotProgress(f'{test_name}/training_info'),
    ]


def compile_opts():
    return {
        'loss': binary_crossentropy,
        'optimizer': Adam(learning_rate=1e-3),
        'metrics': ['accuracy', Precision(), Recall()]
    }


if __name__ == '__main__':
    main()
