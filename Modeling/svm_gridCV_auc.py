from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

X_array = np.load('outputMatrix_userTrimmed.npz')
X = X_array['matrix']

# remove features with low variance (ie more than 80% samples have same value)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_sel = sel.fit_transform(X)
X_sel.shape

y_array = np.load('user_class_array.npz')
y_np = y_array['matrix']
y = y_np.tolist()[0]
y = label_binarize(y, classes=[0, 1, 2, 3, 4])

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.1,
                                                    random_state=9)
model_to_set = OneVsRestClassifier(svm.SVC())

estimator__C = np.logspace(-2, 10, 5)
estimator__gamma = np.logspace(-9, 3, 5)
param_grid = dict(estimator__gamma=estimator__gamma, estimator__C=estimator__C)
cv = KFold(X_train.shape[0], 10)

#cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
grid = GridSearchCV(model_to_set, param_grid=param_grid, cv=cv,
                   n_jobs=-1, pre_dispatch='2*n_jobs', verbose=5, scoring='roc_auc')
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

