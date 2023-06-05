import os.path
import pathlib
from contextlib import redirect_stdout
from os.path import join
from tabulate import tabulate

import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import classification_report, RocCurveDisplay, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import binary_crossentropy

import datasets
from datasets import get_data, get_embedding, pad
from model import get_model_cnn_lstm, get_model_lstm_cnn, get_model_cnn_lstm_bi, get_model_lstm_cnn_bi
from plot_progress import PlotProgress, save_training_info

import warnings

from word2vec import train_word2vec

warnings.filterwarnings('ignore')

batch_size = 128
max_epochs = 10

N_REPEATS = 5
N_SPLITS = 2
kfold = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS)

exp_id = 'one-directional'
headers = [
    "CNN-LSTM",
    "LSTM-CNN",
    # "CNN-LSTM-BI",
    # "LSTM-BI-CNN"
]
models = [
    get_model_cnn_lstm,
    get_model_lstm_cnn,
    # get_model_cnn_lstm_bi,
    # get_model_lstm_cnn_bi
]



def main(together_w2v=False):
    dataset_paths = [
        datasets.twitter_train_file,
        datasets.imdb_file,
    ]
    for d_id, ds in enumerate(dataset_paths):
        ds_name = ds.split(r'/')[1].rstrip('.csv')
        scores = np.zeros((len(models), kfold.get_n_splits()))
        scores_file = f'{exp_id}/{ds_name}/scores.npy'
        if not os.path.exists(scores_file):
            (X, y), (X_test, y_test) = get_data(ds)
            for fold_id, (train, valid) in enumerate(kfold.split(X, y)):
                X_train, y_train, X_val, y_val = X[train], y[train], X[valid], y[valid]
                for m_id, model_type in enumerate(models):
                    model_name = f'{exp_id}/{ds_name}/{headers[m_id].lower()}-f{fold_id}'
                    pathlib.Path(model_name).mkdir(exist_ok=True, parents=True)
                    w2v_path = f'{model_name}/w2v'
                    if not os.path.exists(w2v_path):
                        if together_w2v:
                            w2v = train_word2vec(np.concatenate([X_train, X_val, X_test]))
                        else:
                            w2v = train_word2vec(X_train)
                        w2v.save_word2vec_format(w2v_path, binary=False)
                    else:
                        w2v = KeyedVectors.load_word2vec_format(w2v_path)
                    if not os.path.exists(os.path.join(model_name, 'saved_model.pb')):
                        embed = get_embedding(w2v)
                        model = model_type(embed)
                        X_pad_train, X_pad_val, = pad(X_train, w2v), pad(X_val, w2v),
                        train_model(model, X_pad_train, y_train, X_pad_val, y_val, model_name=model_name)
                    else:
                        print(f'Skipping training {model_name}')
                        model = load_model(model_name)
                    X_pad_test = pad(X_test, w2v)
                    scores[m_id, fold_id] = test(model, X_pad_test, y_test, test_name=model_name)
            np.save(scores_file, scores)
        scores = np.load(scores_file)
        run_stats(f'{exp_id}/{ds_name}', scores)


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    pathlib.Path(model_name).mkdir(exist_ok=True, parents=True)
    model.compile(
        **compile_opts()
    )
    with open(f'{model_name}/summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
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
    # return f1_score(Y, Y_pred_classes)
    return accuracy_score(Y, Y_pred_classes)


def run_stats(ds_path, scores, alfa=0.05, wcox=False):
    n_models = scores.shape[0]
    t_statistic = np.zeros((n_models, n_models))
    p_value = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                if wcox:
                    res = wilcoxon(scores[i], scores[j])
                    t_statistic[i, j], p_value[i, j] = res.statistic, res.pvalue
                else:
                    t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
    advantage = np.zeros((n_models, n_models))
    advantage[t_statistic > 0] = 1
    significance = np.zeros((n_models, n_models))
    significance[p_value < alfa] = 1
    adv_table = significance * advantage
    print_pretty_table(t_statistic, p_value, adv_table, ds_path)


def print_pretty_table(t_statistic, p_value, advantage_table, ds_path):
    names_column = np.array([[n] for n in headers])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".6f")
    adv_table = np.concatenate((names_column, advantage_table), axis=1)
    adv_table = tabulate(adv_table, headers)
    print("t-statistic:\n", t_statistic_table)
    print("\n\np-value:\n", p_value_table)
    print("\n\nadvantage-table:\n", adv_table)
    with open(f'{ds_path}/summary.txt', 'w') as f:
        with redirect_stdout(f):
            print("t-statistic:\n", t_statistic_table)
            print("\n\np-value:\n", p_value_table)
            print("\n\nadvantage-table:\n", adv_table)


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
