# coding=utf-8
# Execució demo de prediccións per als 3 models generats

import numpy as np
from sklearn.metrics import r2_score
from joblib import dump, load
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_dataset(path):
    return pd.read_csv(path, header=0, delimiter=",")


def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val



def load_model():
    atributs_correlacio = ['Rooms', 'Date', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Type(h)', 'Type(t)', 'Type(u)', 'Method(S)', 'Method(SP)', 'Method(PI)', 'Method(VB)', 'Method(SA)', 'big', 'medium', 'small', 'Price']


    models = ['RF', 'DT', 'LR']

    # Amb el d'entrenament
    dataset = load_dataset("../data/little__db_linia_324.csv")
    atributs = dataset.columns.tolist()
    aux = atributs[3:]
    dataset = dataset[aux]
    dataset_norm = standarize(dataset[atributs_correlacio])
    data = dataset_norm.values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    x_train, y_train, x_val, y_val = split_data(x_data, y_data)
    print('Resultats amb el dataset EDA i fent split')
    for m in models:
        regr = load('../models/{}.joblib'.format(m))
        predicted = regr.predict(x_val)
        r2 = round(r2_score(y_val, predicted), 3)
        print('R2 score for model {}: {}'.format(m, r2))
    print('Les R2 score obtingudes són superiors a les obtingudes en les proves perquè les dades amb què es testegen\
            són conegudes pels models tot')

    # Demo
    dataset = load_dataset("../data/demo_dataset.csv")
    atributs = dataset.columns.tolist()
    aux = atributs[4:]
    dataset = dataset[aux]
    dataset_norm = standarize(dataset[atributs_correlacio])
    data = dataset_norm.values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    print('Resultats amb el dataset DEMO')
    for m in models:
        regr = load('../models/{}.joblib'.format(m))
        predicted = regr.predict(x_data)
        r2 = round(r2_score(y_data, predicted), 3)
        print('R2 score for model {}: {}'.format(m, r2))
    print('Les R2 score obtingudes són inferiors a les obtingudes en les proves probablement perquè les dades de DEMO\
     contenen outliers. Especialment perjudicial en el cas del regressor lineal')

load_model()
