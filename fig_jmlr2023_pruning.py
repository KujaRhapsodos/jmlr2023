import os
import time
import pandas as pd
from Learners import SFGDLearner, OptimalStepsizeLearner, LeastSquaresLearner, LassoLearner, CVLearner
from Model import Instantiation1
from Pruners import * 
from dataset_loaders import MNISTLoader, scale_data, add_bias
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = MNISTLoader(digits=[1, 7]).load_valid()
X_train, X_test = scale_data(X_train, X_test)
X_train, X_test = add_bias(X_train, X_test)

TEST = False

RUN_EXPE = True

TABLE_FOLDER = './tables'
TABLE_PATH = TABLE_FOLDER + '/jmlr2023_pruning.tex'
LOG_FOLDER = './results'
LOG_PATH = LOG_FOLDER + '/jmlr2023_pruning.csv'

VERSION = Instantiation1
PRUNER = Method1Pruner
LEARNERS = {SFGDLearner, OptimalStepsizeLearner, LeastSquaresLearner, LassoLearner}
if TEST:
    N_RUNS = 1
    N_ITER = 10
    LEARNER_PARAMS = {'n_iter' : [N_ITER],
                    'regularization' : [0.0000001],
                    'batch_size' : [10],
                    'B' : [1000]}
else:
    N_RUNS = 10
    N_ITER = 1000
    LEARNER_PARAMS = {'n_iter' : [N_ITER],
                    'regularization' : [0.0000001, 0.000001, 0.00001, 0.0001, 0.001],
                    'batch_size' : [50],
                    'B' : [1000]}

def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def check_log_exists():
    return os.path.exists(LOG_PATH)

def save_df(df: pd.DataFrame):
    ensure_folder_exists(LOG_FOLDER)
    df.to_csv(LOG_PATH)
    print('Results dataframe saved to {}'.format(LOG_PATH))

def load_df():
    return pd.read_csv(LOG_PATH, index_col=0)

def run_comparison():
    results_lst = []
    for learner in LEARNERS:
        start = time.time()
        for n in range(N_RUNS):
            results_lst.append(get_one_line(learner=learner, rng=n))
        duration = time.time() - start
        print("Experiment complete for {} in {} seconds.".format(learner.__name__, duration))
    df = pd.DataFrame(results_lst)
    return df

def get_results_df():
    if RUN_EXPE or not check_log_exists():
        df = run_comparison()
    else:
        df = load_df()
    return df

def get_one_line(learner : Learner, rng):
    results = {}
    pruner = Method1Pruner()
    results['algo'] = learner.__name__
    cv = CVLearner(learner, Instantiation1, LEARNER_PARAMS, {}, rng=rng)
    cv.fit(X_train, y_train)
    clf = cv.best_estimator_
    alpha = clf.model.copy()
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results['LS01'] = round(1 - accuracy_score(y_train, train_pred), 3)
    results['LD01'] = round(1 - accuracy_score(y_test, test_pred), 3)
    if learner.__name__ != 'LassoLearner':
        pruner.prune(clf)
    results['%pruning'] = (N_ITER - clf.model.get_n_centers()) / N_ITER * 100
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    results['Lasso+LS01'] = round(1 - accuracy_score(y_train, train_pred), 3)
    results['Lasso+LD01'] = round(1 - accuracy_score(y_test, test_pred), 3)
    beta = clf.model.copy()
    results['norm alpha'] = alpha.norm()
    results['norm beta'] = beta.norm()
    results['norm diff'] = (alpha-beta).norm()
    results['rademacher after lasso'] = clf.rademacher_bound()
    return results

def generate_table(df: pd.DataFrame):
    df = df.replace('Method1Pruner', 'Method 1')
    df = df.replace('Method2Pruner', 'Method 2')
    df = df.replace('Instantiation1', 'I1')
    df = df.replace('SFGDLearner', 'SFGD')
    df = df.replace('OptimalStepsizeLearner', 'Optimal stepsize descent')
    df = df.replace('LeastSquaresLearner', 'Least squares fit')
    df = df.replace('LassoLearner', 'Lasso fit')
    df = df.drop(columns=['norm alpha', 'norm diff', 'rademacher after lasso'])
    table = df.groupby(['algo']).mean()
    table = table.round(3)
    pd.set_option("precision", 3)
    print(table)
    return table

def save_table(table):
    header = ['$\\empirical^{01}$', '$\\risk^{01}$', 
              '\%pruning', 'pruning+$\\empirical^{01}$', 'pruning+$\\risk^{01}$', 
              '$\\normH{\\beta}$']
    ensure_folder_exists(TABLE_FOLDER)
    table.to_latex(TABLE_PATH, column_format='rrrrrrrrrr', header=header, escape=False)
    print('Latex table saved to {}'.format(TABLE_FOLDER))

def run_experiment():
    df = get_results_df()
    save_df(df)
    table = generate_table(df)
    save_table(table)

if __name__ == '__main__':
    start = time.time()
    run_experiment()
    duration = time.time() - start
    print("Experiment time : {} seconds".format(duration))
