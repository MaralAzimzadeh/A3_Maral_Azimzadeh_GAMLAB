from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

data = load_breast_cancer()
X = data.data
y = data.target

'''
data.feature_names:

array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'], dtype='<U23')
'''

kf = KFold(n_splits=5, shuffle=True, random_state=42)

#****************Logistic Regression****************
logistic_params = {
    'C': [0.1, 1, 10],
    'penalty': ['l2']
}
logistic_model = LogisticRegression()
logistic_grid = GridSearchCV(logistic_model, logistic_params, cv=kf, scoring='accuracy')
logistic_grid.fit(X, y)
print('LogisticRegression:')
print('Best Score:', logistic_grid.best_score_)
print('Best Params:', logistic_grid.best_params_)
print('----------------------------------------')

#****************KNeighborsClassifier****************
knn_params = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}
knn_model = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_model, knn_params, cv=kf, scoring='accuracy')
knn_grid.fit(X, y)
print('KNeighborsClassifier:')
print('Best Score:', knn_grid.best_score_)
print('Best Params:', knn_grid.best_params_)
print('----------------------------------------')

#****************DecisionTreeClassifier****************
decision_tree_params = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6]
}
decision_tree_model = DecisionTreeClassifier()
decision_tree_grid = GridSearchCV(decision_tree_model, decision_tree_params, cv=kf, scoring='accuracy')
decision_tree_grid.fit(X, y)
print('DecisionTreeClassifier:')
print('Best Score:', decision_tree_grid.best_score_)
print('Best Params:', decision_tree_grid.best_params_)
print('----------------------------------------')

#****************RandomForestClassifier****************
random_forest_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7]
}
random_forest_model = RandomForestClassifier()
random_forest_grid = GridSearchCV(random_forest_model, random_forest_params, cv=kf, scoring='accuracy')
random_forest_grid.fit(X, y)
print('RandomForestClassifier:')
print('Best Score:', random_forest_grid.best_score_)
print('Best Params:', random_forest_grid.best_params_)
print('----------------------------------------')

#****************Support Vector Classifier (SVC)****************
svc_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
svc_model = SVC()
svc_grid = GridSearchCV(svc_model, svc_params, cv=kf, scoring='accuracy')
svc_grid.fit(X, y)
print('SVC:')
print('Best Score:', svc_grid.best_score_)
print('Best Params:', svc_grid.best_params_)
print('----------------------------------------')

'''
report:

LogisticRegression:
Best Score: 0.9419810588417947
Best Params: {'C': 0.1, 'penalty': 'l2'}
----------------------------------------
KNeighborsClassifier:
Best Score: 0.936686849868033
Best Params: {'n_neighbors': 5, 'weights': 'uniform'}
----------------------------------------
DecisionTreeClassifier:
Best Score: 0.9402577239559076
Best Params: {'max_depth': 5, 'min_samples_split': 2}
----------------------------------------
RandomForestClassifier:
Best Score: 0.9648501785437045
Best Params: {'max_depth': 7, 'n_estimators': 150}
----------------------------------------
SVC:
Best Score: 0.9595249184909175
Best Params: {'C': 10, 'kernel': 'linear'}
----------------------------------------
'''