import os
from contextlib import redirect_stdout
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from scipy.stats import wilcoxon, ttest_rel


def run_stats(ds_path, scores, alfa=0.05, wcox=False, models=None):
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
    print_pretty_table(t_statistic, p_value, adv_table, ds_path, models)


def print_pretty_table(t_statistic, p_value, advantage_table, ds_path, models):
    headers = models.keys()
    names_column = np.array([[n] for n in headers])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".6f")
    adv_table = np.concatenate((names_column, advantage_table), axis=1)
    adv_table = tabulate(adv_table, headers)
    results = f"t-statistic:\n {t_statistic_table}\n\n" \
              f"p-value:\n{p_value_table}\n\n" \
              f"advantage-table:\n{adv_table}"
    print(results)
    with open(f'{ds_path}/stat_scores.txt', 'w') as f:
        with redirect_stdout(f):
            print(results)


def plot_training_metrics(ds_path, metric=None):
    plt.close('all')
    plt.ioff()
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.gca().set_ylim([0.5, 1.0])
    plt.grid()
    for model in os.listdir(ds_path):
        model_path = os.path.join(ds_path, model)
        if os.path.isdir(model_path):
            avg_metric = collect_metrics(model_path, metric=metric)
            plt.plot(avg_metric)

    legends = [os.path.normpath(p) for p in os.listdir(ds_path) if os.path.isdir(join(ds_path, p))]
    plt.legend(legends)
    plt.savefig(join(ds_path, metric))


def calculate_avg_elementwise(metrics):  # needed to do it manually because arrays are different lengths
    max_len = max([len(m) for m in metrics])
    avgs = [0] * max_len
    n_elems = [0] * max_len
    for i, m in enumerate(metrics):
        for j, val in enumerate(m):
            avgs[j] += val
            n_elems[j] += 1
    for j, _ in enumerate(avgs):
        avgs[j] /= n_elems[j]
    return avgs


def collect_metrics(model_path, metric=None):
    metrics = []
    for fold in os.listdir(model_path):
        metrics_path = os.path.join(model_path, fold, 'training_info/history.npy')
        history = np.load(metrics_path, allow_pickle=True).item()
        met = history[metric]
        metrics.append(met)
    return calculate_avg_elementwise(metrics)
