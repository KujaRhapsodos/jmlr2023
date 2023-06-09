import gzip
import os
import urllib

import idx2numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, add_dummy_feature
from sklearn.datasets import load_breast_cancer 

def scale_data(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def add_bias(X_train, X_test):
    X_train = add_dummy_feature(X_train)
    X_test = add_dummy_feature(X_test)
    return X_train, X_test

def sample_batch(X: np.ndarray, y: np.ndarray, batch_size=100, rng=0):
    sample_size, n_features = X.shape
    rng = np.random.default_rng(rng)
    batch_idx = rng.choice(sample_size, size=batch_size, replace=True)
    batch_data = X[batch_idx, :].reshape((batch_size, n_features))
    batch_targets = y[batch_idx]
    return batch_data, batch_targets  

DATA_FOLDER = './datasets/'

class MNISTLoader():
    """Loader for the MNIST dataset.

    Will download the data if necessary.
    """
    def __init__(self, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        self.name = 'mnist'
        if len(digits) < 10:
            for d in digits:
                self.name += str(d)
        self.digits = digits

    def maybe_download(self, file_names):
        WEB_PATH = 'http://yann.lecun.com/exdb/mnist/'
        GZ = '.gz'

        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        for name in file_names:
            if not os.path.exists(name):
                if not os.path.exists(name + GZ):
                    with urllib.request.urlopen(WEB_PATH + name + GZ) as response:
                        file_content = response.read()
                    with open(name + GZ, 'wb') as f:
                        f.write(file_content)
                with gzip.open(name + GZ, 'rb') as f:
                    file_content = f.read()
                with open(name, 'wb') as f:
                    f.write(file_content)
        os.chdir('../')

    def load(self):
        train_data_name = 'train-images-idx3-ubyte'
        train_labels_name = 'train-labels-idx1-ubyte'
        test_data_name = 't10k-images-idx3-ubyte'
        test_labels_name = 't10k-labels-idx1-ubyte'
        file_names = [train_data_name, train_labels_name, test_data_name, test_labels_name]
        self.maybe_download(file_names)

        Xtrain = idx2numpy.convert_from_file(DATA_FOLDER+train_data_name).astype(float)
        ytrain = idx2numpy.convert_from_file(DATA_FOLDER+train_labels_name).astype(int)
        Xtest = idx2numpy.convert_from_file(DATA_FOLDER+test_data_name).astype(float)
        ytest = idx2numpy.convert_from_file(DATA_FOLDER+test_labels_name).astype(int)

        Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
        Xtest = Xtest.reshape((Xtest.shape[0], -1))

        train_indices = np.full(ytrain.shape, False)
        test_indices = np.full(ytest.shape, False)

        for d in self.digits:
            train_indices = np.where((ytrain == d) | train_indices, True, False)
            test_indices = np.where((ytest == d) | test_indices, True, False)

        Xtrain = Xtrain[train_indices,:]
        Xtest = Xtest[test_indices,:]
        ytrain = ytrain[train_indices]
        ytest = ytest[test_indices]

        return Xtrain, Xtest, ytrain, ytest

    def load_valid(self, test_size=0.25, random_state=0):
        Xtrain, _, ytrain, _ = self.load()
        return train_test_split(Xtrain, ytrain, test_size=test_size, random_state=random_state)


class AdultsLoader():
    """Loader for the Adults dataset.

    Will download the data if necessary.

    Preprocesses categorical variables using one-hot encoding.    
    """
    def __init__(self):
        self.name = 'adults'

    def maybe_download(self, train_name, test_name):
        TRAIN_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        TEST_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        
        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        if not os.path.exists(train_name):
            train_df = pd.read_csv(TRAIN_WEB_PATH, header=None)
            train_df.to_csv(train_name)
        if not os.path.exists(test_name):
            test_df = pd.read_csv(TEST_WEB_PATH, header=None, skiprows=1)
            test_df.to_csv(test_name)
        os.chdir('../')
    
    def load(self):
        train_name = 'adults_train.csv'
        test_name = 'adults_test.csv'

        self.maybe_download(train_name, test_name)

        train_df = pd.read_csv(DATA_FOLDER+train_name, index_col=0)
        test_df = pd.read_csv(DATA_FOLDER+test_name, index_col=0)
        n_train = train_df.shape[0]
        full_df = pd.concat((train_df, test_df), axis=0)
        full_df.replace(' <=50K.', ' <=50K', inplace=True)
        full_df.replace(' >50K.', ' >50K', inplace=True)
        full_df = pd.get_dummies(full_df)
        full_df.drop(full_df.columns[len(full_df.columns)-1], axis=1, inplace=True)
        data = full_df.to_numpy()
        Xtrain = data[:n_train,:-1]
        ytrain = data[:n_train, -1]
        Xtest = data[n_train:,:-1]
        ytest = data[n_train:, -1]
        return Xtrain, Xtest, ytrain, ytest

    def load_valid(self, test_size=0.25, random_state=0):
        Xtrain, _, ytrain, _ = self.load()
        return train_test_split(Xtrain, ytrain, test_size=test_size, random_state=random_state)


class SkinSegmentationLoader():
    """Loader for the SkinSegmentation dataset.

    Will download the data if necessary.   
    """
    def __init__(self):
        self.name = 'skin segmentation'
        self.FILE_WEB_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
        self.FILE_NAME = "Skin_NonSkin.txt"

    def maybe_download(self):        
        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

        os.chdir(DATA_FOLDER)
        if not os.path.exists(self.FILE_NAME):
            urllib.request.urlretrieve(self.FILE_WEB_PATH, self.FILE_NAME)
        os.chdir('../')
    
    def load(self, test_size=0.25, random_state=0):
        self.maybe_download()
        df = pd.read_csv(DATA_FOLDER+self.FILE_NAME, sep='\t', header=None)
        data = df.to_numpy()
        X = data[:,:-1]
        y = data[:, -1]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def load_valid(self, test_size=0.25, random_state=0):
        Xtrain, _, ytrain, _ = self.load()
        return train_test_split(Xtrain, ytrain, test_size=test_size, random_state=random_state)

class BreastCancerLoader():
    def __init__(self):
        self.name = 'breast cancer'

    def load(self, test_size=0.25, random_state=0):
        X, y = load_breast_cancer(return_X_y=True)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def load_valid(self, test_size=0.25, random_state=0):
        Xtrain, _, ytrain, _ = self.load()
        return train_test_split(Xtrain, ytrain, test_size=test_size, random_state=random_state)
