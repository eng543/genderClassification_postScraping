from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.1,
                                                    random_state=9)

#criterion__range = ['gini', 'entropy']
depth__range = [2, 4, 6, 8, 10]
estimators__range = [200, 400, 600, 800]
rate__range = [0.1, 0.5, 1, 1.5, 2, 2.5]
param_grid = dict(max_depth=depth__range, n_estimators=estimators__range, learning_rate=rate__range)
cv = KFold(X_train.shape[0], 10)
#cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
grid = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=param_grid, cv=cv,
                   n_jobs=-1, pre_dispatch='2*n_jobs', verbose=5)
                   
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

