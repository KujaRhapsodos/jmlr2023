import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from dataset_loaders import *
from Learners import *
from Model import *
from Pruners import *
from RKS import RKSClassifier

TEST = False

RUN_EXPE = True
GENERATE_TABLES = True
PRINT_BEST_PARAMS = False

RESULTS_FOLDER = './results/'
DATA_FOLDER = './datasets/'
TABLE_PATH = './tables/jmlr2023-sota.tex'
TABLE_WITH_STD_PATH = './tables/jmlr2023-sota-with-std.tex'
FILENAME = 'jmlr2023-sota'
RNG = np.random.default_rng(0)

class Algorithm():
    def __init__(self, clf: Learner, params, name):
        self.clf = clf
        self.params = params
        self.name = name

class TimeTracker():
    def __init__(self, n_total_fits):
        self.start = time.time()
        self.n_completed_fits = 0
        self.n_total_fits = n_total_fits

    def update(self):
        self.n_completed_fits += 1
        time_so_far = time.time() - self.start
        time_remaining = time_so_far / self.n_completed_fits * (self.n_total_fits - self.n_completed_fits)
        print("Fit {} of {} done. Elapsed time : {} hours.".format(self.n_completed_fits, self.n_total_fits, time_so_far/3600))
        print("Estimated time remaining : {} hours.".format(time_remaining/3600))


LEARNERS = [LassoLearner, LeastSquaresLearner]
DATASETS = [
    MNISTLoader(digits=[1, 7]), 
    AdultsLoader(), 
    BreastCancerLoader(), 
    SkinSegmentationLoader(),
    ]
if TEST:
    N_RUNS = 1
    LEARNER_PARAMS = {'n_iter' : [1],
                    'regularization' : [0.00001],
                    'batch_size' : [1]}
    FILENAME = FILENAME + '-test'
    I1_PARAMS = {'target_variance' : [0.9]}
    I3_PARAMS = {'sigma' : [1],
                 'gamma' : [1]}
    AB_PARAMS = {'n_estimators' : [1]}
    RKS_PARAMS = {'n_neurons' : [1], 'regularization' : [1]}
    SVM_PARAMS = {'C' : [1]}
else:
    N_RUNS = 10
    LEARNER_PARAMS = {'n_iter' : [1000],
                    'regularization' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001],
                    'batch_size' : [50],
                    'B' : [1000]}
    I1_PARAMS = {'target_variance' : [0.1, 0.5, 0.9]}
    I3_PARAMS = {'sigma' : [0.01, 0.1, 1],
                 'gamma' : [0.01, 0.1, 1]}
    AB_PARAMS = {'n_estimators' : [10, 25, 50, 100, 150, 200]}
    RKS_PARAMS = {'n_neurons' : [1000], 'regularization' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001]}
    SVM_PARAMS = {'C' : [0.01, 0.1, 1, 10, 100]}
    
MODELS = []
OTHER_ALGOS = []
MODELS.append((Instantiation1, I1_PARAMS))
MODELS.append((Instantiation2, I3_PARAMS))
OTHER_ALGOS.append(Algorithm(AdaBoostClassifier(), AB_PARAMS, "AdaBoost"))  
OTHER_ALGOS.append(Algorithm(RKSClassifier(), RKS_PARAMS, "RKS"))  
OTHER_ALGOS.append(Algorithm(SVC(), SVM_PARAMS, "SVM"))  


def run_one_sfgd_experiment(dataset_loader, learner_class, model_class, model_param_grid, path):
    """Performs one experiment for an SFGD instantiation.

    One experiment means crossvalidating to find the best hyperparameters
    among algorithm.params, then calculating various metrics on the
    final model. Results are saved in the results folder.
    """
    results = {}
    results['dataset'] = [dataset_loader.name]
    results['algorithm'] = [learner_class.__name__+' w/ '+model_class.__name__]

    X_train, X_test, y_train, y_test = dataset_loader.load()
    X_train, X_test = scale_data(X_train, X_test)
    X_train, X_test = add_bias(X_train, X_test)
    
    cv = CVLearner(learner_class, model_class, LEARNER_PARAMS, model_param_grid, rng=RNG)
    cv.fit(X_train, y_train)
    results['fit time'] = cv.refit_time_

    clf = cv.best_estimator_
    for key in cv.best_learner_params_:
        results[key] = cv.best_learner_params_[key]
    for key in cv.best_model_params_:
        results[key] = cv.best_model_params_[key]

    start_pred = time.time()
    train_pred = clf.predict(X_train)
    results['train pred time'] = [time.time() - start_pred]
    test_pred = clf.predict(X_test)

    results['train 01'] = [round(1 - accuracy_score(y_train, train_pred), 3)]
    results['test 01'] = [round(1 - accuracy_score(y_test, test_pred), 3)]
    results['train loss'] = [clf.calculate_loss(X_train, y_train)]
    results['test loss'] = [clf.calculate_loss(X_test, y_test)]
    results['rademacher'] = [clf.rademacher_bound()]
    results['norm'] = [clf.model.norm()]

    df = pd.DataFrame.from_dict(results)
    if os.path.exists(path):
        previous_df = pd.read_csv(path, index_col=0)
        df = pd.concat((previous_df, df), sort=True)
        df.index = pd.Index(np.arange(len(df.index)))
    df.to_csv(path)
    return

def run_other_experiment(dataset_loader, algorithm, path):
    """Run experiments for algorithms other than an SFGD instantiation.

    Simple crossvalidation with training and test errors.
    Saves results in the results folder.
    """
    results = {}
    results['dataset'] = [dataset_loader.name]
    results['algorithm'] = [algorithm.name]

    X_train, X_test, y_train, y_test = dataset_loader.load()
    cv = GridSearchCV(algorithm.clf, algorithm.params, cv=5, verbose=0, n_jobs=-1)
    cv.fit(X_train, y_train)
    results['fit time'] = cv.refit_time_
    
    clf = cv.best_estimator_
    for key in cv.best_params_:
        results[key] = cv.best_params_[key]

    start_pred = time.time()
    train_pred = clf.predict(X_train)
    results['train pred time'] = [time.time() - start_pred]
    test_pred = clf.predict(X_test)

    results['train 01'] = [round(1 - accuracy_score(y_train, train_pred), 3)]
    results['test 01'] = [round(1 - accuracy_score(y_test, test_pred), 3)]

    try:
        results['train loss'] = [clf.mse(X_train, y_train)]
        results['test loss'] = [clf.mse(X_test, y_test)]
    except:
        pass

    df = pd.DataFrame.from_dict(results)
    if os.path.exists(path):
        previous_df = pd.read_csv(path, index_col=0)
        df = pd.concat((previous_df, df), sort=True)
        df.index = pd.Index(np.arange(len(df.index)))
    df.to_csv(path)
    return

def launch_experiments(results_path):
    # SVM does not require multiple fits, thus the awkward formula for number of fits
    n_total_fits = (1 + N_RUNS * (len(MODELS) * len(LEARNERS) + (len(OTHER_ALGOS) - 1))) * len(DATASETS)
    timetracker = TimeTracker(n_total_fits)
    for dataset in DATASETS:
        for learner_class in LEARNERS:
            for (model_class, params_grid) in MODELS:
                for _ in range(N_RUNS): 
                    run_one_sfgd_experiment(dataset, learner_class, model_class, params_grid, results_path)
                    timetracker.update()
        for algo in OTHER_ALGOS:
            if algo.clf.__class__.__name__ == 'SVC':
                n_runs = 1
            else:
                n_runs = N_RUNS
            for _ in range(n_runs): 
                run_other_experiment(dataset, algo, results_path)
                timetracker.update()

def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_results_path():
    n = 0
    ensure_folder_exists(RESULTS_FOLDER)
    file_name = RESULTS_FOLDER + FILENAME + str(n) + '.csv'
    while os.path.exists(file_name):
        n += 1
        file_name = RESULTS_FOLDER + FILENAME + str(n) + '.csv'
    return file_name

def get_latest_results_path():
    n = 0
    ensure_folder_exists(RESULTS_FOLDER)
    while os.path.exists(RESULTS_FOLDER + FILENAME + str(n+1) + '.csv'):
        n += 1
    return RESULTS_FOLDER + FILENAME + str(n) + '.csv'

def get_df_from_results_path(path):
    df = get_raw_df(path)
    return clean_df(df)

def get_raw_df(path):
    return pd.read_csv(path, index_col=0)

def clean_df(df):
    df_slice = df[['algorithm', 'dataset',
               'train loss', 'test loss', 'rademacher',
               'train 01', 'test 01', 'fit time']]
    df_slice = df_slice.replace('LassoLearner', 'Lasso fit', regex=True)
    df_slice = df_slice.replace('LeastSquaresLearner', 'LS fit', regex=True)
    df_slice = df_slice.replace('Instantiation1', 'I1', regex=True)
    df_slice = df_slice.replace('Instantiation3', 'I2', regex=True)
    return df_slice

def get_table_from_df(df):
    table = df.groupby(['dataset', 'algorithm']).mean()
    table = table.round(3)
    pd.set_option("precision", 3)
    return clean_table(table)

def clean_table(table):
    table['rademacher'] = clean_column(table['rademacher'], 1, 1000)
    table = table.replace(np.nan, '', regex=True)
    table = table.replace('nan', '', regex=True)
    table.index.names = ['dataset', 'algo']
    return table

def clean_column(c, rnd, max_value):
    indices = c >= max_value
    new_c = c[indices]
    new_c = new_c.apply(np.log10)
    new_c = new_c.apply(np.floor)
    new_c = new_c.astype(int)
    new_c = new_c.astype(str)
    for i in np.arange(len(new_c)):
        new_c[i] = '>10' + '\\textsuperscript{' + str(new_c[i]) + '}'
    c = c.round(rnd)
    c = c.astype(str)
    c[indices] = new_c
    return c

def get_std_table_from_df(df):
    table = df.groupby(['dataset', 'algorithm']).std()
    table = table.round(3)  
    pd.set_option("precision", 3)
    return clean_table(table)

def get_table_with_std_from_table_and_std(table, std_table):
    table_with_std = table.astype(str) + ' ± ' + std_table.astype(str)
    table_with_std = table_with_std.replace(' ± ', '', regex=False)
    return table_with_std

def save_table_to_latex(table, path):
    header = ['$\\empirical$', '$\\risk$', 
          '$\\rademacher$',
          '$\\empirical^{01}$', '$\\risk^{01}$',
          'training time (s)']
    table.to_latex(path, column_format='rrrrrrrrr', header=header, escape=False)

def generate_and_save_tables(results_path):
    df = get_df_from_results_path(results_path)
    table = get_table_from_df(df)
    std_table = get_std_table_from_df(df)
    table_with_std = get_table_with_std_from_table_and_std(table, std_table)
    save_table_to_latex(table, TABLE_PATH)
    save_table_to_latex(table_with_std, TABLE_WITH_STD_PATH)
    print("tables saved to ./tables/")

def print_best_params(path):
    df = get_raw_df(path)
    print(df[['dataset', 'algorithm', 'target_variance', 'regularization']])

if __name__ == '__main__':
    if RUN_EXPE:
        results_path = get_results_path()
        launch_experiments(results_path)
        print("results saved in " + results_path)
    if GENERATE_TABLES:
        results_path = get_latest_results_path()
        generate_and_save_tables(results_path)
    if PRINT_BEST_PARAMS:
        results_path = get_latest_results_path()
        print_best_params(results_path)
