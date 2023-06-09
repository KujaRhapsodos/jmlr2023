import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import time

from dataset_loaders import MNISTLoader, scale_data, add_bias
from Model import *
from Learners import *
from Loss import MSE

from fig_jmlr2023_sota import ensure_folder_exists

import visuals # Must be at the end of the imports for some reason


TEST = False

RUN_EXPE = True
GENERATE_FIGURES = True

N_RUNS = 10
RNG = np.random.default_rng(0)

RESULTS_FOLDER = './results/'
FIGURES_FOLDER = './figures/'
FILENAME = 'jmlr2023-algo-compar'

X_train, X_test, y_train, y_test = MNISTLoader(digits=[1, 7]).load_valid()
X_train, X_test = scale_data(X_train, X_test)
X_train, X_test = add_bias(X_train, X_test)

if TEST:
    N_ITERS = [int(x) for x in np.logspace(1, 2, num=5)]
    REGULARIZATION = [0.00001]
    BATCH_SIZE = 10
    FILENAME += '-test'
else:
    N_ITERS = [int(x) for x in np.logspace(math.log10(100), math.log10(4000), num=10)]
    REGULARIZATION = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
    BATCH_SIZE = 50

def do_expe_for_one_learner(learner_class: Learner, **learner_params):
    results_df = pd.DataFrame({})

    learner_params = {**learner_params, 'regularization' : REGULARIZATION}
    model_params = {}
    for seed in range(N_RUNS):
        start = time.time() 
        for n_iter in N_ITERS:
            partial_results = {}
            cv = CVLearner(learner_class, Instantiation1, {**learner_params, 'n_iter' : [n_iter]}, 
                            model_params, rng=RNG)
            cv.fit(X_train, y_train)

            partial_results['algorithm'] = learner_class.__name__
            partial_results['T'] = n_iter
            partial_results['Training time'] = cv.refit_time_
            partial_results['Training MSE'] = cv.best_estimator_.calculate_loss(X_train, y_train)
            partial_results['Training error'] = 1 - cv.score(X_train, y_train)
            partial_results['Test MSE'] = cv.best_estimator_.calculate_loss(X_test, y_test)
            partial_results['Test error'] = 1 - cv.score(X_test, y_test)
            partial_results_df = pd.DataFrame(partial_results, index=[0])
            results_df = pd.concat((results_df, partial_results_df))
        elapsed = time.time() - start
        print("Finished testing seed {} of {} for {} in {} seconds.".format(seed+1, N_RUNS, learner_class.__name__, elapsed))

    return results_df

def get_results_df_path():
    ensure_folder_exists(RESULTS_FOLDER)
    return RESULTS_FOLDER+FILENAME+'.csv'

def get_figure_path():
    ensure_folder_exists(FIGURES_FOLDER)
    return FIGURES_FOLDER+FILENAME

def get_results_df():
    df = get_raw_df()
    return clean_df(df)

def get_raw_df():
    return pd.read_csv(get_results_df_path(), index_col=0)

def clean_df(df: pd.DataFrame):
    df.replace('SFGDLearner', 'SFGD', inplace=True)
    df.replace('LeastSquaresLearner', 'Least squares fit', inplace=True)
    df.replace('LassoLearner', 'Lasso fit', inplace=True)
    df.replace('OptimalStepsizeLearner', 'Optimal stepsize descent', inplace=True)
    grouped = df.groupby(by=['algorithm', 'T'], as_index=False).mean()
    return grouped

def expe(use_batch=True):
    # these lists must be the same length
    learners = [SFGDLearner, OptimalStepsizeLearner, LeastSquaresLearner, LassoLearner]
    learner_params = [{'batch_size' : [BATCH_SIZE], 'loss' : [MSE()], 'B' : [1000]}, 
                      {'use_batch' : [use_batch], 'batch_size' : [BATCH_SIZE], 'loss' : [MSE()]},
                      {}, 
                      {}]

    results_df = pd.DataFrame({})

    for L, params in zip(learners, learner_params):
        partial_results_df = do_expe_for_one_learner(L, **params)
        results_df = pd.concat((results_df, partial_results_df), ignore_index=True)

    results_df.to_csv(get_results_df_path())

def make_figures():

    results_df = get_results_df()
    plt.figure(figsize=(4,4))
    for algo in results_df['algorithm'].unique():
        df = results_df.loc[results_df['algorithm'] == algo]
        x = df['T']
        y = df['Training error']
        plt.plot(x, y, label=algo)
    plt.xlabel('T')
    plt.ylabel('Training error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_figure_path() + '-train_01.pdf')
    plt.clf()

    for algo in results_df['algorithm'].unique():
        df = results_df.loc[results_df['algorithm'] == algo]
        x = df['T']
        y = df['Test error']
        plt.plot(x, y, label=algo)
    plt.xlabel('T')
    plt.ylabel('Test error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_figure_path() + '-test_01.pdf')
    plt.clf()

    for algo in results_df['algorithm'].unique():
        df = results_df.loc[results_df['algorithm'] == algo]
        x = df['T']
        y = df['Training time']
        plt.plot(x, y, label=algo)
    plt.xlabel('T')
    plt.ylabel('Training time (s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_figure_path() + '-train_time.pdf')
    plt.clf()

if __name__ == '__main__':
    if RUN_EXPE:
        expe(use_batch=True)
    if GENERATE_FIGURES:
        make_figures()

