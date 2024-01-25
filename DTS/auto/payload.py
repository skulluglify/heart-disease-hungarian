#!/usr/bin/env python
# -*- encoding: UTF-8 -*-

import os
import math
import joblib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import secrets
import string

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    r2_score,
    precision_score, 
    classification_report, 
    confusion_matrix, 
    mean_squared_error, 
    mean_absolute_error,
)

from pandas.core.series import Series
from sklearn.base import is_regressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

from typing import Any, Literal, Mapping, Sequence, Tuple


Options = Literal['mean', 'std']


cpus = os.cpu_count()
cmap_skyblue = plt.get_cmap('Blues')
mpl_dark2_palette = sns.palettes.mpl_palette('Dark2')
mpl_gray_palette = sns.palettes.mpl_palette('gray')

grid_cv_models = [
    ('RandomForestClassifier', lambda : GridSearchCV(RandomForestClassifier(), dict(
            n_estimators=[10, 50, 100, 200],
            max_depth=[3, 5, 10],
            criterion=['gini', 'entropy'],
        ),
        scoring="accuracy",
        cv=5,
    )),

    ('LinearRegression', lambda : GridSearchCV(LinearRegression(), dict(
            fit_intercept= [True],
            copy_X= [True],
            positive= [True],
        ),
        scoring="r2",
        cv=5,
    )),

    ('LogisticRegression', lambda : GridSearchCV(LogisticRegression(), dict(
            solver=['liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            C=np.logspace(-3, 3, 7),
        ),
        scoring="r2",
        cv=5,
    )),

    ('DecisionTreeClassifier', lambda : GridSearchCV(DecisionTreeClassifier(), dict(
            max_depth=[3, 5, 10],
            criterion=['gini', 'entropy'],
            splitter=['best', 'random'],
        ),
        scoring="accuracy",
        cv=5,
    )),

    ('DecisionTreeRegressor', lambda : GridSearchCV(DecisionTreeRegressor(), dict(
            max_depth=[3, 5, 10],
            criterion=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            splitter=['best', 'random'],
        ),
        scoring="r2",
        cv=5,
    )),

    ('KNeighborsClassifier', lambda : GridSearchCV(KNeighborsClassifier(), dict(
            n_neighbors=[3, 5, 7, 9],
            weights=['uniform', 'distance'],
            metric=['euclidean', 'manhattan', 'minkowski'],
        ),
        scoring="accuracy",
        cv=5,
    )),

    ('XGBClassifier', lambda : GridSearchCV(XGBClassifier(), dict(
            n_estimators=[10, 50, 100, 200],
            max_depth=[3, 5, 10],
            learning_rate=[0.01, 0.1, 0.2, 0.3],
        ),
        scoring="accuracy",
        cv=5,
    )),
]

models = [
    ('KNeighborsClassifier', lambda : KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='minkowski')),
    ('RandomForestClassifier', lambda : RandomForestClassifier(n_estimators=100, random_state=42)),
    ('LinearRegression', lambda : LinearRegression(fit_intercept=True, copy_X=False, positive=False)),
    ('LogisticRegression', lambda : LogisticRegression(solver='liblinear')),
    ('DecisionTreeClassifier', lambda : DecisionTreeClassifier(max_depth=5, criterion='gini', splitter='best')),
    ('DecisionTreeRegressor', lambda : DecisionTreeRegressor(max_depth=5, criterion='squared_error', splitter='best')),
    ('XGBClassifier', lambda : XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=42)),
    ('SVC', lambda : SVC()),
]

def show_data_part(df: pd.DataFrame):
    msno.matrix(df)
    plt.show()


def auto_mapping_data(df: pd.DataFrame) -> Mapping[str, Mapping[str, int]]:
    mapping_data = {}
    for column in df.columns:
        df_column = df[column]
        
        if df_column.dtype not in (np.dtype('O'),):
            continue
        
        df_column_unique = df_column.unique()
        mapping_dict = dict([(v, i) for i, v in enumerate(df_column_unique)])
        
        df_column.replace(mapping_dict, inplace=True)
        mapping_data[column] = mapping_dict

    return mapping_data


def auto_set_or_drop(df: pd.DataFrame, percent: float = 0.5, option: Options | None = 'std'):
    calc = df.isnull().sum() / df.sum()
    test = calc > percent
    data = test.to_dict()

    for key, upper in data.items():
        
        # drop it, too much nan!
        if upper:
            df.drop(key, axis=1, inplace=True)
            continue

        # make it noisy!
        df_column = df[key]

        if option == 'mean':
            df_column.fillna(df_column.mean(), inplace=True)

        elif option == 'std':
            df_column.fillna(df_column.std(), inplace=True)

        else:
            df.dropna(inplace=True)
    
    # drop constant values
    df.drop(df.columns[df.nunique() == 1], axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)


def make_xy(df: pd.DataFrame, column_name_target: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
    x = df.drop(column_name_target, axis=1).to_numpy()
    y = df[column_name_target].to_numpy()
    
    return x, y


Choices = Sequence[Tuple[str, Tuple[np.ndarray, np.ndarray], Sequence[Any]]]

def make_choices(x: np.ndarray, y: np.ndarray) -> Choices:
    
    # oversampling
    smote = SMOTE(random_state=42)
    adasyn = ADASYN(random_state=42)

    # setup
    x_smote, y_smote = smote.fit_resample(x, y)
    x_adasyn, y_adasyn = adasyn.fit_resample(x, y)

    # undersampling
    nearmiss = NearMiss(n_neighbors=3)

    # setup
    x_nearmiss, y_nearmiss = nearmiss.fit_resample(x, y)
    
    # origin
    scaler_norm = MinMaxScaler()
    x_norm = scaler_norm.fit_transform(x)

    # over-sampling
    scaler_smote_norm = MinMaxScaler()
    x_smote_norm = scaler_smote_norm.fit_transform(x_smote)

    scaler_adasyn_norm = MinMaxScaler()
    x_adasyn_norm = scaler_adasyn_norm.fit_transform(x_adasyn)

    # under-sampling
    scaler_nearmiss_norm = MinMaxScaler()
    x_nearmiss_norm = scaler_nearmiss_norm.fit_transform(x_nearmiss)
        
    # origin
    scaler_std = StandardScaler()
    x_std = scaler_std.fit_transform(x)

    # over-sampling
    scaler_smote_std = StandardScaler()
    x_smote_std = scaler_smote_std.fit_transform(x_smote)

    scaler_adasyn_std = StandardScaler()
    x_adasyn_std = scaler_adasyn_std.fit_transform(x_adasyn)

    # under-sampling
    scaler_nearmiss_std = StandardScaler()
    x_nearmiss_std = scaler_nearmiss_std.fit_transform(x_nearmiss)
    
    # origin
    pca = PCA(n_components=2, random_state=0)
    x_pca = pca.fit_transform(x)

    pca_norm = PCA(n_components=2, random_state=0)
    x_pca_norm = pca_norm.fit_transform(x_norm)

    pca_std = PCA(n_components=2, random_state=0)
    x_pca_std = pca_std.fit_transform(x_std)

    # over-sampling
    pca_smote = PCA(n_components=2, random_state=0)
    x_smote_pca = pca_smote.fit_transform(x_smote)

    pca_smote_norm = PCA(n_components=2, random_state=0)
    x_smote_pca_norm = pca_smote_norm.fit_transform(x_smote_norm)

    pca_smote_std = PCA(n_components=2, random_state=0)
    x_smote_pca_std = pca_smote_std.fit_transform(x_smote_std)

    pca_adasyn = PCA(n_components=2, random_state=0)
    x_adasyn_pca = pca_adasyn.fit_transform(x_adasyn)

    pca_adasyn_norm = PCA(n_components=2, random_state=0)
    x_adasyn_pca_norm = pca_adasyn_norm.fit_transform(x_adasyn_norm)

    pca_adasyn_std = PCA(n_components=2, random_state=0)
    x_adasyn_pca_std = pca_adasyn_std.fit_transform(x_adasyn_std)

    # under-sampling
    pca_nearmiss = PCA(n_components=2, random_state=0)
    x_nearmiss_pca = pca_nearmiss.fit_transform(x_nearmiss)

    pca_nearmiss_norm = PCA(n_components=2, random_state=0)
    x_nearmiss_pca_norm = pca_nearmiss_norm.fit_transform(x_nearmiss_norm)

    pca_nearmiss_std = PCA(n_components=2, random_state=0)
    x_nearmiss_pca_std = pca_nearmiss_std.fit_transform(x_nearmiss_std)
    
    choices = [

        # origin
        ('x', (x, y), []),
        ('x_pca', (x_pca, y), [pca]),
        ('x_pca_norm', (x_pca_norm, y), [scaler_norm, pca_norm]),
        ('x_pca_std', (x_pca_std, y), [scaler_std, pca_std]),
        ('x_norm', (x_norm, y), [scaler_norm]),
        ('x_std', (x_std, y), [scaler_std]),

        # smote
        ('x_smote', (x_smote, y_smote), [smote]),
        ('x_smote_pca', (x_smote_pca, y_smote), [smote, pca_smote]),
        ('x_smote_pca_norm', (x_smote_pca_norm, y_smote), [smote, scaler_smote_norm, pca_smote_norm]),
        ('x_smote_pca_std', (x_smote_pca_std, y_smote), [smote, scaler_smote_std, pca_smote_std]),
        ('x_smote_norm', (x_smote_norm, y_smote), [smote, scaler_smote_norm]),
        ('x_smote_std', (x_smote_std, y_smote), [smote, scaler_smote_std]),

        # adasyn
        ('x_adasyn', (x_adasyn, y_adasyn), [adasyn]),
        ('x_adasyn_pca', (x_adasyn_pca, y_adasyn), [adasyn, pca_adasyn]),
        ('x_adasyn_pca_norm', (x_adasyn_pca_norm, y_adasyn), [adasyn, scaler_adasyn_norm, pca_adasyn_norm]),
        ('x_adasyn_pca_std', (x_adasyn_pca_std, y_adasyn), [adasyn, scaler_adasyn_std, pca_adasyn_std]),
        ('x_adasyn_norm', (x_adasyn_norm, y_adasyn), [adasyn, scaler_adasyn_norm]),
        ('x_adasyn_std', (x_adasyn_std, y_adasyn), [adasyn, scaler_adasyn_std]),

        # nearmiss
        ('x_nearmiss', (x_nearmiss, y_nearmiss), [nearmiss]),
        ('x_nearmiss_pca', (x_nearmiss_pca, y_nearmiss), [nearmiss, pca_nearmiss]),
        ('x_nearmiss_pca_norm', (x_nearmiss_pca_norm, y_nearmiss), [nearmiss, scaler_nearmiss_norm, pca_nearmiss_norm]),
        ('x_nearmiss_pca_std', (x_nearmiss_pca_std, y_nearmiss), [nearmiss, scaler_nearmiss_std, pca_nearmiss_std]),
        ('x_nearmiss_norm', (x_nearmiss_norm, y_nearmiss), [nearmiss, scaler_nearmiss_norm]),
        ('x_nearmiss_std', (x_nearmiss_std, y_nearmiss), [nearmiss, scaler_nearmiss_std]),
    ]
    
    return choices


def show_comparison_from_choices(choices: Choices):
    x_smote, y_smote = None, None
    x_adasyn, y_adasyn = None, None
    x_nearmiss, y_nearmiss = None, None
    x, y = None, None
    
    for (choice_name, (choice_x, choice_y), _) in choices:
        if choice_name == 'x':
            x, y = choice_x, choice_y
            continue
        
        if choice_name == 'x_smote':
            x_smote, y_smote = choice_x, choice_y
            continue
        
        if choice_name == 'x_adasyn':
            x_adasyn, y_adasyn = choice_x, choice_y
            continue
        
        if choice_name == 'x_nearmiss':
            x_nearmiss, y_nearmiss = choice_x, choice_y
            continue

    if x is None or y is None \
        or x_smote is None or y_smote is None \
            or x_adasyn is None or y_adasyn is None \
                or x_nearmiss is None or y_nearmiss is None:

        raise Exception('something goes wrong, i can feel it')

    df_y = pd.DataFrame(data=y)
    df_y_max = math.floor(df_y.value_counts().max() * 1.1)

    fig, ax = plt.subplots(3, 2)

    plt.subplot(3, 2, 1)
    df_y_view = df_y.value_counts()
    df_y_axes = df_y_view.plot(kind='bar', figsize=(8, 12), color=mpl_gray_palette)
    df_y_axes.set_ylim([0, df_y_max])
    plt.title('Target before over sampling with SMOTE')
    plt.xticks(rotation=0)

    plt.subplot(3, 2, 2)
    df_y_smote = pd.DataFrame(data=y_smote)
    df_y_smote_view = df_y_smote.value_counts()
    df_y_smote_axes = df_y_smote_view.plot(kind='bar', figsize=(8, 12), color=mpl_gray_palette)
    df_y_smote_axes.set_ylim([0, df_y_max])
    plt.title('Target after over sampling with SMOTE')
    plt.xticks(rotation=0)

    plt.subplot(3, 2, 3)
    df_y_view = df_y.value_counts()
    df_y_axes = df_y_view.plot(kind='bar', figsize=(8, 12), color=mpl_gray_palette)
    df_y_axes.set_ylim([0, df_y_max])
    plt.title('Target before over sampling with ADASYN')
    plt.xticks(rotation=0)

    plt.subplot(3, 2, 4)
    df_y_adasyn = pd.DataFrame(data=y_adasyn)

    df_y_adasyn_view = df_y_adasyn.value_counts()
    df_y_adasyn_axes = df_y_adasyn_view.plot(kind='bar', figsize=(8, 12), color=mpl_gray_palette)
    df_y_adasyn_axes.set_ylim([0, df_y_max])
    plt.title('Target after over sampling with ADASYN')
    plt.xticks(rotation=0)

    plt.subplot(3, 2, 5)
    df_y_view = df_y.value_counts()
    df_y_axes = df_y_view.plot(kind='bar', figsize=(8, 12), color=mpl_gray_palette)
    df_y_axes.set_ylim([0, df_y_max])
    plt.title('Target before under sampling with NearMiss')
    plt.xticks(rotation=0)

    plt.subplot(3, 2, 6)
    df_y_nearmiss = pd.DataFrame(data=y_nearmiss)

    df_y_nearmiss_view = df_y_nearmiss.value_counts()
    df_y_nearmiss_axes = df_y_nearmiss_view.plot(kind='bar', figsize=(8, 12), color=mpl_gray_palette)
    df_y_nearmiss_axes.set_ylim([0, df_y_max])
    plt.title('Target after under sampling with NearMiss')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


def find_pos_by_idx(idx: int, rows: int, cols: int) -> Tuple[int, int]:
    z_idx = idx + 1
    k_row = 0
    k_col = 0
    
    for row in range(rows):
        stopped = 0
        
        x_row = row + 1
        s_row = row * cols
        
        for col in range(cols):
            x_col = col + 1
            y_idx = s_row + x_col
            
            if z_idx >= y_idx:
                k_row = x_row
                k_col = x_col
                continue
                
            stopped = 1
            break
        
        if stopped == 1:
            break
    
    return k_row, k_col


def show_data_pca_from_choices(choices: Choices):
    x_pca_data_all = [(choice_name, *choice_xy) for (choice_name, choice_xy, _) in choices if 'pca' in choice_name]

    x_pca_data_all_count = len(x_pca_data_all)
    k = math.sqrt(x_pca_data_all_count)
    rows, cols = math.floor(k), math.ceil(k)
    # fig, ax = plt.subplots(rows, cols, figsize=(rows * 8, cols * 8))


    pca_df_mul = []
    for idx, (x_pca_name, x_pca_data, y_data) in enumerate(x_pca_data_all):
        k_row, k_col = find_pos_by_idx(idx, rows, cols)
        print(k_row, k_col, x_pca_name)
        
        pca_df = pd.DataFrame(data=x_pca_data, columns=['PC1', 'PC2'])
        pca_df['name'] = x_pca_name
        pca_df['target'] = y_data
        pca_df['k_row'] = k_row
        pca_df['k_col'] = k_col
        
        pca_df_mul.append(pca_df)

    pca_df = pd.concat(pca_df_mul)
    # plt.figure(layout='constrained')
    sns.lmplot(data=pca_df, x='PC1', y='PC2', row='k_row', col='k_col', hue='target', facet_kws=dict(sharex=False, sharey=False))
    # sns.lmplot(data=pca_df, x='PC1', y='PC2', row='k_row', col='k_col', hue='target', fit_reg=False, legend=True, height=4, facet_kws=dict(sharex=False, sharey=False))
    plt.show()


def minmax(v, v_min=0.0, v_max=1.0, p_nan=True, p_none=True):
  if p_none:
    if v is None:
      return 0.0
  
  if p_nan:
    if math.isnan(v):
      return 0.0
  
  return min(max(v, v_min), v_max)


EvaluationMode = Literal['f1', 'r2']

def evaluation(y_test, y_pred, mode: EvaluationMode = 'f1'):

    if mode == 'f1':
        return dict(
            accuracy=minmax(accuracy_score(y_test, y_pred, normalize=True)),
            recall=minmax(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            precision=minmax(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            f1_score=minmax(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        )
  
    if mode == 'r2':
        return dict(
            mse=minmax(mean_squared_error(y_test, y_pred)),
            mae=minmax(mean_absolute_error(y_test, y_pred)),
            r2_score=minmax(r2_score(y_test, y_pred)),
        )
    
    return dict()


def pprint(data, start = '', end = '\r\n', output = False, tab = '\t'):
    if isinstance(data, Mapping):
        temp = '|' + end
    
        for k, v in data.items():
            temp += start + k + ' = ' + pprint(v, tab + start, end, output=True) + end
    
        if not output:
            print(temp)
      
        return temp

    if not isinstance(data, str):
        if isinstance(data, Sequence):
            temp = '|' + end
        
            for k, v in enumerate(data):
                temp += start + str(k) + ' = ' + pprint(v, tab + start, end, output=True) + end
        
            if not output:
                print(temp)
            
            return temp

    if not output:
        print(data)
    
    return str(data)


Results = Sequence[Mapping[str, Any]]

def train_models(models, choices: Choices, test_size=0.2, random_state=42, n_splits=1) -> Results:
    results = []

    for model_idx, (model_name, model_cb) in enumerate(models):
        print(model_name, '#' + str(model_idx))

        model = model_cb()
        regression = is_regressor(model)

        model_results = []
        for choice_idx, (choice_name, (x, y), preprocessing) in enumerate(choices):
            print('--', choice_name, '#' + str(choice_idx))

            # splitting data training 80%, and testing 20%
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

            # KFold from 80% of x_train from train_test_split

            # using kfold instead of train_test_split
            model = model_cb()
            kfold_scores = np.zeros(shape=(n_splits,), dtype=np.float64)

            # using kfold must be n_splits greater than or equal of 2
            if n_splits > 1:
                kf = KFold(n_splits)

                for kf_idx, (train_index, test_index) in enumerate(kf.split(x_train)):
                    kf_x_train, kf_x_test = None, None
                    kf_y_train, kf_y_test = None, None

                    if isinstance(x_train, Series):
                        kf_x_train, kf_x_test = x_train.iloc[train_index], x_train.iloc[test_index]

                    elif isinstance(x_train, np.ndarray):
                        kf_x_train, kf_x_test = x_train[train_index], x_train[test_index]

                    else:
                        raise Exception('type data x is not valid')

                    if isinstance(y_train, Series):
                        kf_y_train, kf_y_test = y_train.iloc[train_index], y_train.iloc[test_index]

                    elif isinstance(y_train, np.ndarray):
                        kf_y_train, kf_y_test = y_train[train_index], y_train[test_index]

                    else:
                        raise Exception('type data y is not valid')

                    model.fit(kf_x_train, kf_y_train)
                    y_pred = model.predict(kf_x_test)

                    scores = evaluation(kf_y_test, y_pred, mode='f1' if not regression else 'r2')
                    score = scores.get('f1_score' if not regression else 'r2_score')
                    if score is None:
                        score = scores.get('f1_score') or scores.get('r2_score')

                    score = minmax(score)

                    kfold_scores[kf_idx] = score
                    print('KFold', '#' + str(kf_idx), 'score=' + str(score))

            elif n_splits == 1:
                model.fit(x_train, y_train)

            else:
                raise Exception('n_splits must be greater than 0')

            # final evaluation
            kfold_score = np.mean(kfold_scores)

            # testing model, prediction
            y_pred = model.predict(x_test)

            #* FIX y_pred not same a y_test (indexes)

            # make it same as type y_pred and y_test
            y_pred = y_pred.round().astype(y_test.dtype)

            # evaluation
            scores = evaluation(y_test, y_pred, mode='f1' if not regression else 'r2')
            pprint(scores, tab='    ')

            # append data results
            model_results.append(dict(
                model=model,
                name=choice_name,
                y_pred=y_pred, y_test=y_test,    # matrix confusion (required)
                preprocessing=preprocessing,
                kfold_score=kfold_score,
                scores=scores,
            ))

            print('--')
        print()

        results.append(dict(
            name=model_name,
            results=model_results,
        ))

    return results


def get_comparison_from_results(results: Results) -> pd.DataFrame:
    comparison_values = []

    for result_idx, result in enumerate(results):
        model = result.get('model')
        model_name = result.get('name')
        model_results = result.get('results')
        print(model_name, '#' + str(result_idx))

        regression = is_regressor(model)

        for choice_idx, choice in enumerate(model_results):
            choice_name = choice.get('name')
            scores = choice.get('scores')
            name = model_name + '_' + choice_name

            print('--', name, '#' + str(choice_idx))
            score = scores.get('f1_score' if not regression else 'r2_score')
            if score is None:
                score = scores.get('f1_score') or scores.get('r2_score')

            score = minmax(score)

            comparison_values.append((name, score))

        print(end='\r\n\r\n')

    comparison = pd.DataFrame(comparison_values, columns=['Model', 'Accuracy'])
    comparison = comparison.sort_values(by='Accuracy', ascending=False)
    comparison.head()

    return comparison


def gen_short_name(chars: int = 8):
    return "".join(secrets.choice(string.digits + string.ascii_letters) for _ in range(chars))


def secret_names(names, uniques = []):
    unique_names = []

    for _ in range(len(names)):
        name = gen_short_name()

        while True:
            if name in uniques:
                name = gen_short_name()
                continue

            break

        unique_names.append(name)
    return unique_names


def plot_comparison(comparison: pd.DataFrame, k: int = 12):

    n = comparison.values.shape[0]
    p = n / k
    v = math.sqrt(p)

    rows, cols = math.floor(v), math.ceil(v)
    idx = 0

    #* FIX p great than rows and cols
    if rows * cols < p:
        rows = cols

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 10, rows * 12))
    uniques = []
    refs = []

    for i in range(0, n, k):
        names = comparison['Model'].values[i:i+k]
        scores = comparison['Accuracy'].values[i:i+k]

        unique_names = secret_names(names, uniques)

        ref = dict(zip(unique_names, names))
        refs.append(ref)
        pprint(refs, tab='    ')

        plt.subplot(rows, cols, idx + 1)
        bars = plt.bar(unique_names, scores, color=mpl_dark2_palette)
        idx += 1

        # static y-score
        plt.ylim([0, 1])

        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('Evaluation')
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

    return refs


def make_rank_from_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    rank_values = [(idx + 1, name.split('_')[0], comparison['Accuracy'].values[idx]) for idx, name in enumerate(comparison['Model'].unique())]
    rank = pd.DataFrame(rank_values, columns=['Rank', 'Model', 'Accuracy'])
    rank.drop_duplicates(subset='Model', inplace=True)
    return rank


def best_model_from_results(results: Results) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], float]:
    """
    define:\r\n
    -> best_model_data, model_selected, choice_selected, max_score\r\n
    """
    choice_selected = None
    model_selected = None
    max_score = 0.0

    for model_idx, model in enumerate(results):
        model_results = model.get('results')

        for choice_idx, choice in enumerate(model_results):
            scores = choice.get('scores')
            model_fit = choice.get('model')
            regression = is_regressor(model_fit)
            score = scores.get('f1_score' if not regression else 'r2_score')
            if score is None:
                score = scores.get('f1_score') or scores.get('r2_score')

            score = minmax(score)

            if max_score < score:
                max_score = score
                model_selected = model
                choice_selected = choice

    if model_selected is None or choice_selected is None:
        raise Exception('couldn\'t get best model from results')

    best_model_data = dict(
        name=model_selected.get('name') + '_' + choice_selected.get('name'),
        model=choice_selected.get('model'),
        use_smote=choice_selected.get('name').startswith('x_smote'),
        use_adasyn=choice_selected.get('name').startswith('x_adasyn'),
        use_nearmiss=choice_selected.get('name').startswith('x_nearmiss'),
        preprocessing=choice_selected.get('preprocessing'),
        scores=choice_selected.get('scores'),
    )

    return best_model_data, model_selected, choice_selected, max_score


def show_confusion_matrix_from_best_model(model_selected: Mapping[str, Any], choice_selected: Mapping[str, Any]):
    if model_selected is None or choice_selected is None:
        raise Exception('couldn\'t get best model from results')

    name = model_selected.get('name')
    choice_name = choice_selected.get('name')

    scores = choice_selected.get('scores')
    y_pred = choice_selected.get('y_pred')
    y_test = choice_selected.get('y_test')

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_skyblue)
    plt.title('Confusion Matrix')
    plt.show()


def save_best_model(best_model_data: Mapping[str, Any], filename: str = 'best_model_data.jbl'):
    with open(filename, 'wb') as fstream:
        joblib.dump(best_model_data, fstream)


def op(df: pd.DataFrame, column_name_target: str) -> Tuple[Mapping[str, Any], pd.DataFrame]:

    refs = auto_mapping_data(df)

    show_data_part(df)
    auto_set_or_drop(df)
    show_data_part(df)

    x, y = make_xy(df, column_name_target)
    choices = make_choices(x, y)
    show_comparison_from_choices(choices)
    show_data_pca_from_choices(choices)

    results = train_models(models, choices, test_size=0.2, n_splits=4)
    comparison = get_comparison_from_results(results)
    plot_comparison(comparison)

    rank = make_rank_from_comparison(comparison)

    best_model_data, model_selected, choice_selected, max_score = best_model_from_results(results)

    show_confusion_matrix_from_best_model(model_selected, choice_selected)

    best_model_data['refs'] = refs
    pprint(best_model_data)

    return rank, best_model_data


def banner():
    context = '' +\
    '   ____   _   _  __  __   ____   ____      ____    ____ __  __     ____ __  __ ____   ____  _  ____  ' +\
    '  / () \ | |_| ||  \/  | / () \ | _) \    / () \  (_ (_`\ \/ /    (_ (_`\ \/ // () \ | ===|| |/ () \ ' +\
    ' /__/\__\|_| |_||_|\/|_|/__/\__\|____/   /__/\__\.__)__) |__|    .__)__) |__|/__/\__\|__|  |_|\____/=' +\
    '                                                                                                     ' +\
    'Copyright by Ahmad Asy Syafiq, Payload Script For Data Science 2023/2024                             '
    
    print(context)
    return context
