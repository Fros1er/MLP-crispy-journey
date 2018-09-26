from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#--algorithms--
#SVM
from sklearn.svm import SVC
#LR
from sklearn.linear_model import LogisticRegression
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#KNN
from sklearn.neighbors import KNeighborsClassifier
#NB
from sklearn.naive_bayes import GaussianNB
#CART
from sklearn.tree import DecisionTreeClassifier
#AB
from sklearn.ensemble import AdaBoostClassifier
#GBM
from sklearn.ensemble import GradientBoostingClassifier
#RFR
from sklearn.ensemble import RandomForestClassifier
#ETR
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings("ignore")

#vars
scoring='accuracy'

def KNN_params(X_train, Y_train, num_folds, seed):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
    model = KNeighborsClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=rescaledX, y=Y_train)
    print('best: %s using %s' % (grid_result.best_score_, grid_result.best_params_))
    cv_results = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
    for mean, std, param in cv_results:
        print('mean: %f std: %f with %r' % (mean, std, param))

def SVM_params(X_train, Y_train, num_folds, seed):
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train).astype(float)
    param_grid = {}
    param_grid['C'] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    param_grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    model = SVC()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=rescaledX, y=Y_train)
    print('best: %s using %s' % (grid_result.best_score_, grid_result.best_params_))
    cv_results = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
    for mean, std, param in cv_results:
        print('mean: %f std: %f with %r' % (mean, std, param))

#def GBM_params()

def test_standardscaler(X_train, Y_train, num_folds, seed, boxing=False):
    pipelines = {}
    pipelines['ScalerLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])
    pipelines['ScalerLDA'] = Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])
    pipelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])
    pipelines['ScalerNB'] = Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])
    pipelines['ScalerCART'] = Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])
    pipelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])
    
    kfold = KFold(n_splits=num_folds, random_state=seed)
    results = []
    for key in pipelines:
        result = cross_val_score(pipelines[key], X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(result)
        print('%s: mean=%.3f std=%.3f' % (key, result.mean(), result.std()))
    print('\n')
    if boxing:
        box(results, pipelines)

def test_algorithm(X_train, Y_train, num_folds, seed, boxing=False):
    model = {}
    model['LR'] = LogisticRegression()
    model['LDA'] = LinearDiscriminantAnalysis()
    model['KNN'] = KNeighborsClassifier()
    model['NB'] = GaussianNB()
    model['CART'] = DecisionTreeClassifier()
    model['SVM'] = SVC()

    kfold = KFold(n_splits=num_folds, random_state=seed)
    results = []
    for key in model:
        result = cross_val_score(model[key], X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(result)
        print('%s: mean=%.3f std=%.3f' % (key, result.mean(), result.std()))
    if boxing:
        box(results, model)

def test_standardscaler_ensemble(X_train, Y_train, num_folds, seed, boxing=False):
    pipelines = {}
    pipelines['ScalerAB'] = Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostClassifier())])
    pipelines['ScalerGBM'] = Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingClassifier())])
    pipelines['ScalerRFR'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestClassifier())])
    pipelines['ScalerETR'] = Pipeline([('Scaler', StandardScaler()), ('ETR', ExtraTreesClassifier())])

    kfold = KFold(n_splits=num_folds, random_state=seed)
    results = []
    for key in pipelines:
        result = cross_val_score(pipelines[key], X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(result)
        print('%s: mean=%.3f std=%.3f' % (key, result.mean(), result.std()))
    print('\n')
    if boxing:
        box(results, pipelines)

def test_ensemble(X_train, Y_train, num_folds, seed, boxing=False):
    ensembles = {}
    ensembles['AB'] = AdaBoostClassifier()
    ensembles['GBM'] = GradientBoostingClassifier()
    ensembles['RFR'] = RandomForestClassifier()
    ensembles['ETR'] = ExtraTreesClassifier()

    kfold = KFold(n_splits=num_folds, random_state=seed)
    results = []
    for key in ensembles:
        result = cross_val_score(ensembles[key], X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(result)
        print('%s: mean=%.3f std=%.3f' % (key, result.mean(), result.std()))
    if boxing:
        box(results, ensembles)


#关系矩阵图
def interface_matrix(a):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(a.corr(), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    plt.show()

#箱线图
def box(a,models):
    fig = plt.figure()
    fig.suptitle('QuQ')
    ax = fig.add_subplot(111)
    plt.boxplot(a)
    ax.set_xticklabels(models.keys())
    plt.show()
