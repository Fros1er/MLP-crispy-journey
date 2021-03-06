from pandas import read_csv
from functions import get_result, SVC, GBM_params, KNN_params, SVM_params, test
from sklearn.model_selection import train_test_split

names = [i for i in range(0,60)]
names.append('class')
dataset = read_csv('sonar.csv', names=names)

array = dataset.values
X = array[:, 0:60]
Y = array[:, 60]
num_folds = 10
seed = 7
validation_size = 0.33
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)



'''
GBM_params(X_train, Y_train, num_folds, seed)
KNN_params(X_train, Y_train, num_folds, seed)
SVM_params(X_train, Y_train, num_folds, seed)
'''

get_result(SVC(C=1.5, kernel='rbf'), X_train, Y_train, X_validation, Y_validation)
